package org.tensorflow.demo;

import android.app.Activity;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.RectF;
import android.graphics.Typeface;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.SystemClock;
import android.util.Log;
import android.util.Size;
import android.util.TypedValue;
import android.view.Surface;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Toast;
import android.support.v7.app.AppCompatActivity;
import org.tensorflow.demo.R;
import org.tensorflow.demo.env.BorderedText;
import org.tensorflow.demo.env.ImageUtils;
import org.tensorflow.demo.env.Logger;
import org.tensorflow.demo.tracking.MultiBoxTracker;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.LinkedList;
import java.util.List;
import java.util.Vector;

import static android.content.ContentValues.TAG;

public class ImageDetectionActivity extends AppCompatActivity {
    private static final Logger LOGGER = new Logger();
    static final boolean AF_NEURAL_NETWORK_SHOW_OBJECT = true;
    static final boolean AF_NEURAL_NETWORK_SHOW_OBJECT_TRACKING = true;
    private static final boolean MAINTAIN_ASPECT = false;

    //模型路径设置
    private static final int TF_OD_API_INPUT_SIZE = 300;
    private static final String TF_OD_API_MODEL_FILE =
            "file:///android_asset/frozen_inference_graph.pb";
    private static final String TF_OD_API_LABELS_FILE = "file:///android_asset/label_map.txt";

    private enum DetectorMode {
        TF_OD_API;
    }
    //检测模式
    private static final DetectorMode MODE = DetectorMode.TF_OD_API;
    //检测结果置信度门限
    private static final float MINIMUM_CONFIDENCE_TF_OD_API = 0.5f;

    private static final Size DESIRED_PREVIEW_SIZE = new Size(640, 480);
    private static final boolean SAVE_PREVIEW_BITMAP = false;
    private static final float TEXT_SIZE_DIP = 10;

    private Classifier detector;
    private Integer sensorOrientation;

    private long lastProcessingTimeMs;
    private Bitmap rgbFrameBitmap = null;
    private Bitmap croppedBitmap = null;
    private Bitmap cropCopyBitmap = null;

    private boolean computingDetection = false;

    private long timestamp = 0;

    OverlayView trackingOverlay;

    private Matrix frameToCropTransform;
    private Matrix cropToFrameTransform;

    private MultiBoxTracker tracker;

    private byte[] luminanceCopy;

    private BorderedText borderedText;

    private String IMG_NAME = "digger4.jpg";
    private Button detect_img;
    private ImageView imageViewResult;
    //frame data
    boolean readyToProcessFrame = false;
    Bitmap currentFrameBmp;
    int rotation = 0;
    //detector
    protected int previewWidth = 0;
    protected int previewHeight = 0;
    private boolean debug = false;

    //handler
    private Handler handler;
    private HandlerThread handlerThread;

    //kernel;
    static int frameIndex = 1;



    public Bitmap getBitmapFromAssetsFolder(String fileName)
    {
        Bitmap bitmap = null;
        try
        {
            InputStream istr=getAssets().open(fileName);
            bitmap= BitmapFactory.decodeStream(istr);
        }
        catch (IOException e1)
        {
            // TODO Auto-generated catch block
            //e1.printStackTrace();
            System.out.println("Error: " + e1);
            System.exit(0);
        }
        return bitmap;
    }


    protected int getLuminanceStride()
    {
        return previewWidth;  //yRowStride
    }
    protected synchronized void runInBackground(final Runnable r)
    {
        if (handler != null) {
            handler.post(r);
        }
    }

    public void addCallback(final OverlayView.DrawCallback callback)
    {
        final OverlayView overlay = (OverlayView) findViewById(R.id.debug_overlay);
        if (overlay != null) {
            overlay.addCallback(callback);
        }
    }

    public void requestRender()
    {
        final OverlayView overlay = (OverlayView) findViewById(R.id.debug_overlay);
        if (overlay != null) {
            overlay.postInvalidate();
        }
    }

    protected int getScreenOrientation()
    {
        switch (getWindowManager().getDefaultDisplay().getRotation())
        {
            case Surface.ROTATION_270:
                return 270;
            case Surface.ROTATION_180:
                return 180;
            case Surface.ROTATION_90:
                return 90;
            default:
                return 0;
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_image_detection);
        Bitmap imageBitmap = getBitmapFromAssetsFolder(IMG_NAME);
        int imageWidth = imageBitmap.getWidth();
        int imageHeight = imageBitmap.getHeight();
        imageViewResult = (ImageView)findViewById(R.id.imageView);
        imageViewResult.setImageBitmap(imageBitmap);
//        detect_img = (Button) findViewById(R.id.detect_img);
        currentFrameBmp = imageBitmap;
        readyToProcessFrame = true;

        initialiseDetector(imageWidth, imageHeight, rotation);
        executeKernel();

    }

    public void initialiseDetector(final int imageWidth, final int imageHeight, final int rotation)
    {
        final float textSizePx = TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
        borderedText = new BorderedText(textSizePx);
        borderedText.setTypeface(Typeface.MONOSPACE);

        tracker = new MultiBoxTracker(this);

        int cropSize = TF_OD_API_INPUT_SIZE;

        try
        {
            detector = TensorFlowObjectDetectionAPIModel.create(getAssets(), TF_OD_API_MODEL_FILE, TF_OD_API_LABELS_FILE, TF_OD_API_INPUT_SIZE);
            cropSize = TF_OD_API_INPUT_SIZE;
        }
        catch (final IOException e)
        {
            LOGGER.e("Exception initializing classifier!", e);
            Toast toast = Toast.makeText(getApplicationContext(), "Classifier could not be initialized", Toast.LENGTH_SHORT);
            toast.show();
            finish();
        }

        previewWidth = imageWidth;
        previewHeight = imageHeight;

        sensorOrientation = rotation - getScreenOrientation();
        LOGGER.i("Camera orientation relative to screen canvas: %d", sensorOrientation);

        LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
        rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Bitmap.Config.ARGB_8888);
        croppedBitmap = Bitmap.createBitmap(cropSize, cropSize, Bitmap.Config.ARGB_8888);

        frameToCropTransform = ImageUtils.getTransformationMatrix(previewWidth, previewHeight, cropSize, cropSize, sensorOrientation, MAINTAIN_ASPECT);

        cropToFrameTransform = new Matrix();
        frameToCropTransform.invert(cropToFrameTransform);
        if(AF_NEURAL_NETWORK_SHOW_OBJECT_TRACKING)
        {
            //object detection/tracking display code;
            trackingOverlay = (OverlayView) findViewById(R.id.tracking_overlay);
            trackingOverlay.addCallback(
                    new OverlayView.DrawCallback() {
                        @Override
                        public void drawCallback(final Canvas canvas) {
                            LOGGER.i("drawCallback1");
                            tracker.draw(canvas);
                        }
                    });

            addCallback(
                    new OverlayView.DrawCallback() {
                        @Override
                        public void drawCallback(final Canvas canvas) {
                            LOGGER.i("drawCallback2");

                            final Bitmap copy = cropCopyBitmap;
                            if (copy == null) {
                                return;
                            }

                            final int backgroundColor = Color.argb(100, 0, 0, 0);
                            canvas.drawColor(backgroundColor);

                            final Matrix matrix = new Matrix();
                            final float scaleFactor = 2;
                            matrix.postScale(scaleFactor, scaleFactor);
                            matrix.postTranslate(
                                    canvas.getWidth() - copy.getWidth() * scaleFactor,
                                    canvas.getHeight() - copy.getHeight() * scaleFactor);
                            canvas.drawBitmap(copy, matrix, new Paint());

                            final Vector<String> lines = new Vector<String>();
                            if (detector != null) {
                                final String statString = detector.getStatString();
                                final String[] statLines = statString.split("\n");
                                for (final String line : statLines) {
                                    lines.add(line);
                                }
                            }
                            lines.add("");

                            lines.add("Frame: " + previewWidth + "x" + previewHeight);
                            lines.add("Crop: " + copy.getWidth() + "x" + copy.getHeight());
                            lines.add("View: " + canvas.getWidth() + "x" + canvas.getHeight());
                            lines.add("Rotation: " + sensorOrientation);
                            lines.add("Inference time: " + lastProcessingTimeMs + "ms");

                            borderedText.drawLines(canvas, 10, canvas.getHeight() - 10, lines);
                        }
                    });
        }

    }

    private void executeKernel()
    {
		/*
		Bitmap currentFrameBmpTemp = currentFrameBmp;

		byte[] currentFrameLuminance = calculateLuminanceMap(currentFrameBmpTemp);
		processImage(currentFrameLuminance, currentFrameBmpTemp);
		Log.i(TAG, "processImage");
		*/

        final Handler handlerKernel = new Handler();
        final Runnable r = new Runnable()
        {
            //@Override
            public void run()
            {
                handlerKernel.postDelayed(this, 120);	//executing object detection every second

                if(readyToProcessFrame && !computingDetection)
                {
                    Bitmap currentFrameBmpTemp = currentFrameBmp;

                    byte[] currentFrameLuminance = calculateLuminanceMap(currentFrameBmpTemp);
                    processImage(currentFrameLuminance, currentFrameBmpTemp);
                    Log.i(TAG, "processImage");
                }
            }
        };
        handlerKernel.postDelayed(r, 66);
    }

    @Override
    protected void onResume() {
        super.onResume();
        //detector;
        handlerThread = new HandlerThread("inference");
        handlerThread.start();
        handler = new Handler(handlerThread.getLooper());
    }

    @Override
    protected void onPause()
    {
        super.onPause();

        //detector;
        if (!isFinishing()) {
            LOGGER.d("Requesting finish");
            finish();
        }
        handlerThread.quitSafely();
        try
        {
            handlerThread.join();
            handlerThread = null;
            handler = null;
        }
        catch (final InterruptedException e)
        {
            LOGGER.e(e, "Exception!");
        }
    }


    private byte[] calculateLuminanceMap(Bitmap bitmap)
    {
        int R = 0; int G = 0; int B = 0;
        int height = bitmap.getHeight();
        int width = bitmap.getWidth();
        byte[] luminanceMap = new byte[width * height];
        int[] pixels = new int[width * height];
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height);
        for (int i = 0; i < pixels.length; i++)
        {
            int color = pixels[i];
            R += Color.red(color);
            G += Color.green(color);
            B += Color.blue(color);
            byte luminance = (byte)((R+G+B)/3);
            luminanceMap[i] = luminance;
        }
        return luminanceMap;
    }

    protected void processImage(byte[] originalLuminanceTemp, Bitmap currentFrameBmpTemp)
    {
        ++timestamp;
        final long currTimestamp = timestamp;

        tracker.onFrame(
                previewWidth,
                previewHeight,
                getLuminanceStride(),
                sensorOrientation,
                originalLuminanceTemp,
                timestamp);
        if(AF_NEURAL_NETWORK_SHOW_OBJECT_TRACKING)
        {
            trackingOverlay.postInvalidate();
        }

        LOGGER.i("processImage2");
        if(computingDetection)
        {
            LOGGER.i("computingDetection - skip frame");
            //skip frame
            return;
        }
        computingDetection = true;
        LOGGER.i("Preparing image " + currTimestamp + " for detection in bg thread.");

        rgbFrameBitmap = currentFrameBmpTemp;

        if (luminanceCopy == null)
        {
            luminanceCopy = new byte[originalLuminanceTemp.length];
        }
        System.arraycopy(originalLuminanceTemp, 0, luminanceCopy, 0, originalLuminanceTemp.length);

        final Canvas canvas = new Canvas(croppedBitmap);
        canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);
        // For examining the actual TF input.
        if (SAVE_PREVIEW_BITMAP)
        {
            ImageUtils.saveBitmap(croppedBitmap);
        }

        runInBackground(
                new Runnable()
                {
                    @Override
                    public void run()
                    {
                        LOGGER.i("Running detection on image " + currTimestamp);
                        final long startTime = SystemClock.uptimeMillis();
                        final List<Classifier.Recognition> results = detector.recognizeImage(croppedBitmap);
                        lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;

                        cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);
                        final Canvas canvas = new Canvas(cropCopyBitmap);
                        final Paint paint = new Paint();
                        paint.setColor(Color.RED);
                        paint.setStyle(Paint.Style.STROKE);
                        paint.setStrokeWidth(2.0f);

                        float minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
//                        switch (MODE)
//                        {
//                            case TF_OD_API:
//                                minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
//                                break;
//                        }

                        final List<Classifier.Recognition> mappedRecognitions = new LinkedList<Classifier.Recognition>();


                        for (final Classifier.Recognition result : results)
                        {
                            final RectF location = result.getLocation();
                            LOGGER.i("object found");

                            if (location != null && result.getConfidence() >= minimumConfidence)
                            {
                                LOGGER.i("passedConfidence");
                                String toastResult = "classes:"+result.getTitle()+"location:"+location+"confidence:"+result.getConfidence();
                                Toast.makeText(getApplicationContext(), toastResult, Toast.LENGTH_LONG).show();
                                canvas.drawRect(location, paint);

                                cropToFrameTransform.mapRect(location);
                                result.setLocation(location);
                                mappedRecognitions.add(result);

                            }
                        }

                        frameIndex++;

                        if(AF_NEURAL_NETWORK_SHOW_OBJECT)
                        {
                            tracker.trackResults(mappedRecognitions, luminanceCopy, currTimestamp);//this also adds a box around found objects
                            trackingOverlay.postInvalidate();
                            requestRender();

                        }

                        computingDetection = false;
                    }
                });
    }

}
