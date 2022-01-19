/*
 * Copyright 2020 Google LLC. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.mlkit.vision.demo.java.posedetector;

import android.content.Context;
import androidx.annotation.NonNull;
import androidx.annotation.RequiresApi;

import android.os.Build;
import android.os.SystemClock;
import android.os.Trace;
import android.util.Log;
import com.google.android.gms.tasks.Task;
import com.google.android.odml.image.MlImage;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.demo.GraphicOverlay;
import com.google.mlkit.vision.demo.java.VisionProcessorBase;
import com.google.mlkit.vision.demo.java.posedetector.classification.CSVFile;
import com.google.mlkit.vision.demo.java.posedetector.classification.DeepLearningClassification;
import com.google.mlkit.vision.demo.java.posedetector.classification.PoseClassifier;
import com.google.mlkit.vision.demo.java.posedetector.classification.PoseClassifierProcessor;
import com.google.mlkit.vision.demo.java.posedetector.classification.PoseSample;
import com.google.mlkit.vision.demo.java.posedetector.classification.RepetitionCounter;
import com.google.mlkit.vision.pose.Pose;
import com.google.mlkit.vision.pose.PoseDetection;
import com.google.mlkit.vision.pose.PoseDetector;
import com.google.mlkit.vision.pose.PoseDetectorOptionsBase;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

/** A processor to run pose detector. */
public class PoseDetectorProcessor
    extends VisionProcessorBase<PoseDetectorProcessor.PoseWithClassification> {
  private static final String TAG = "PoseDetectorProcessor";

  private final PoseDetector detector;

  private final boolean showInFrameLikelihood;
  private final boolean visualizeZ;
  private final boolean rescaleZForVisualization;
  private final boolean runClassification;
  private final boolean isStreamMode;
  private final Context context;
  private final Executor classificationExecutor;
  private final boolean deepLearning = true;
  private long previousTime = SystemClock.elapsedRealtime();

  private PoseClassifierProcessor poseClassifierProcessor;
  /** Internal class to hold Pose and classification results. */
  public static class PoseWithClassification {
    private final Pose pose;
    private final List<String> classificationResult;

    public PoseWithClassification(Pose pose, List<String> classificationResult) {
      this.pose = pose;
      this.classificationResult = classificationResult;
    }

    public Pose getPose() {
      return pose;
    }

    public List<String> getClassificationResult() {
      return classificationResult;
    }
  }

  @RequiresApi(api = Build.VERSION_CODES.N)
  public PoseDetectorProcessor(
      Context context,
      PoseDetectorOptionsBase options,
      boolean showInFrameLikelihood,
      boolean visualizeZ,
      boolean rescaleZForVisualization,
      boolean runClassification,
      boolean isStreamMode) {
    super(context);
    this.showInFrameLikelihood = showInFrameLikelihood;
    this.visualizeZ = visualizeZ;
    this.rescaleZForVisualization = rescaleZForVisualization;
    detector = PoseDetection.getClient(options);
    this.runClassification = runClassification;
    this.isStreamMode = isStreamMode;
    this.context = context;
    classificationExecutor = Executors.newSingleThreadExecutor();
    testData();
  }

  @RequiresApi(api = Build.VERSION_CODES.N)
  private void testData() {
    List<PoseSample> poseSamples = new ArrayList<>();
    try {
      BufferedReader reader = new BufferedReader(
              new InputStreamReader(context.getAssets().open("yoga_test_poses_csvs_out.csv")));
      String csvLine = reader.readLine();
      while (csvLine != null) {
        // If line is not a valid {@link PoseSample}, we'll get null and skip adding to the list.
        PoseSample poseSample = PoseSample.getPoseSample(csvLine, ",");
        if (poseSample != null) {
          poseSamples.add(poseSample);
        }
        csvLine = reader.readLine();
      }
    } catch (IOException e) {
      Log.e(TAG, "Error when loading pose samples.\n" + e);
    }
    for(PoseSample sample : poseSamples){
//      System.out.println("points: " + sample.getLandmarks().stream().findFirst());
//      System.out.println("should be: " + sample.getClassName() );
      DeepLearningClassification deepLearning = new DeepLearningClassification();
      deepLearning.classifySample(sample, context);
//      System.out.println("result sample: " + deepLearning.classifySample(sample, context));

    }
  }

  @Override
  public void stop() {
    super.stop();
    detector.close();
  }

  @Override
  protected Task<PoseWithClassification> detectInImage(InputImage image) {
    return detector
        .process(image)
        .continueWith(
            classificationExecutor,
            task -> {
              Pose pose = task.getResult();
              List<String> classificationResult = new ArrayList<>();
              if(deepLearning){
                DeepLearningClassification deepLearning = new DeepLearningClassification();
                deepLearning.classify(pose, context);
                return new PoseWithClassification(pose, deepLearning.classify(pose, context));
              }
              if (runClassification) {
                System.out.println("run classification");
                if (poseClassifierProcessor == null) {
                  poseClassifierProcessor = new PoseClassifierProcessor(context, isStreamMode);
                }
                classificationResult = poseClassifierProcessor.getPoseResult(pose);
              }
              return new PoseWithClassification(pose, classificationResult);
            });
  }

  @Override
  protected Task<PoseWithClassification> detectInImage(MlImage image, long frameStartMs) {
    return detector
        .process(image)
        .continueWith(
            classificationExecutor,
            task -> {
              Pose pose = task.getResult();
              List<String> classificationResult = new ArrayList<>();
              if(deepLearning){
                Trace.beginSection("deeplearning classifier");
                DeepLearningClassification deepLearning = new DeepLearningClassification();
                List<String> results = deepLearning.classify(pose,context);
                Trace.endSection();
                return new PoseWithClassification(pose, results);

              }
              if (runClassification) {
                if (poseClassifierProcessor == null) {
                  poseClassifierProcessor = new PoseClassifierProcessor(context, isStreamMode);
                }
                classificationResult = poseClassifierProcessor.getPoseResult(pose);
              }
              return new PoseWithClassification(pose, classificationResult);
            });
  }

  @Override
  protected void onSuccess(
      @NonNull PoseWithClassification poseWithClassification,
      @NonNull GraphicOverlay graphicOverlay) {
    graphicOverlay.add(
        new PoseGraphic(
            graphicOverlay,
            poseWithClassification.pose,
            showInFrameLikelihood,
            visualizeZ,
            rescaleZForVisualization,
            poseWithClassification.classificationResult));
  }

  @Override
  protected void onFailure(@NonNull Exception e) {
    Log.e(TAG, "Pose detection failed!", e);
  }

  @Override
  protected boolean isMlImageEnabled(Context context) {
    System.out.println("isMLimageEnabled true");
    // Use MlImage in Pose Detection by default, change it to OFF to switch to InputImage.
    return true;
  }
}
