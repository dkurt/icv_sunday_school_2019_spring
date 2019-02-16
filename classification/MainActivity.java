package org.opencv.samples.icvclassificationdemo;

import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;

import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgproc.Imgproc;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.ServerSocket;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;

public class MainActivity extends AppCompatActivity implements CvCameraViewListener2 {

  @Override
  public void onResume() {
    super.onResume();
    System.loadLibrary("opencv_java4");
    mOpenCvCameraView.enableView();
  }

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_main);
    // Set up camera listener.
    mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.CameraView);
    mOpenCvCameraView.setVisibility(CameraBridgeViewBase.VISIBLE);
    mOpenCvCameraView.setCvCameraViewListener(this);
  }

  @Override
  public void onCameraViewStarted(int width, int height) {
    String weights = "/sdcard/Android/data/org.opencv.samples.icvclassificationdemo/squeezenet_v1.1.caffemodel";
    String config = "/sdcard/Android/data/org.opencv.samples.icvclassificationdemo/squeezenet_v1.1.prototxt";
    net = Dnn.readNet(weights, config);
    classes = readClasses();

    new Thread() {
      public void run() {
        try {
          ServerSocket serverSocket = new ServerSocket(SERVER_SOCKET_PORT);
          clientSocketOut = serverSocket.accept().getOutputStream();
        } catch (IOException e) {
          e.printStackTrace();
        }
      }
    }.start();
  }

  @Override
  public void onCameraViewStopped() {

  }

  public Mat onCameraFrame(CvCameraViewFrame inputFrame) {
    Mat frame = inputFrame.rgba();
    Mat frameBGR = new Mat();
    Imgproc.cvtColor(frame, frameBGR, Imgproc.COLOR_RGBA2BGR);

    Mat blob = Dnn.blobFromImage(frameBGR, 1.0, new Size(227, 227), new Scalar(104, 117, 123));
    net.setInput(blob);
    Mat out = net.forward();

    Core.MinMaxLocResult loc = Core.minMaxLoc(out);
    if (loc.maxVal > 0.5)
    {
      String label = classes.get((int)loc.maxLoc.x);
      Imgproc.putText(frame, label, new Point(50, 50), Imgproc.FONT_HERSHEY_SIMPLEX,
                      1.0, new Scalar(0, 255, 0), 2);
    }

    if (clientSocketOut != null) {
      sendImg(frame);
    }

    return frame;
  }

  static private ArrayList<String> readClasses() {
    ArrayList<String> classes = new ArrayList();
    try {
      FileInputStream fstream = new FileInputStream("/sdcard/Android/data/org.opencv.samples.icvclassificationdemo/classes.txt");
      BufferedReader reader = new BufferedReader(new InputStreamReader(fstream));

      String line;
      while ((line = reader.readLine()) != null) {
        classes.add(line);
      }

      fstream.close();
    } catch (IOException e) {
      e.printStackTrace();
    }
    return classes;
  }

  private void sendImg(Mat img) {
    int rows = img.rows();
    int cols = img.cols();
    int channels = img.channels();
    byte[] imgData = new byte[rows * cols * channels];
    img.get(0, 0, imgData);

    byte[] data = ByteBuffer.allocate(4 * 3)
            .order(ByteOrder.nativeOrder())
            .putInt(rows).putInt(cols).putInt(channels)
            .array();
    try {
      clientSocketOut.write(data);
      clientSocketOut.write(imgData);
      clientSocketOut.flush();
    } catch (IOException e) {
      e.printStackTrace();
    }
  }

  private CameraBridgeViewBase mOpenCvCameraView;
  private Net net;
  private ArrayList<String> classes;
  private final int SERVER_SOCKET_PORT = 43656;
  private OutputStream clientSocketOut = null;
}
