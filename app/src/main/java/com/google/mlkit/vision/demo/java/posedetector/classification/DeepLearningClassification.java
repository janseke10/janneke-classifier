package com.google.mlkit.vision.demo.java.posedetector.classification;

import static com.google.mlkit.vision.demo.java.posedetector.classification.PoseEmbedding.getPoseEmbedding;

import android.content.Context;
import android.content.res.AssetFileDescriptor;

import android.os.Environment;

import com.google.mlkit.vision.common.PointF3D;

import com.google.mlkit.vision.demo.ml.FinalModel3;

import com.google.mlkit.vision.pose.Pose;
import com.google.mlkit.vision.pose.PoseLandmark;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;


import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;


import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;

import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class DeepLearningClassification {

    public static final String CSV_FILE_NAME = "embedding.csv";

    private ByteBuffer loadModelFile(Context context) throws IOException {
        AssetFileDescriptor fileDescriptor = context.getAssets().openFd("withWeights.tflite");
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }


    public List<String> withInterpreter(PoseSample sample, Context context){
        System.out.println("Okay here the withInterpreter model starts!");
//      File file = new File("withWeights.tflite");

        ClassificationResult result = new ClassificationResult();

        // Creates inputs for reference.
        TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 99}, DataType.FLOAT32);
        TensorBuffer output = TensorBuffer.createFixedSize(new int[]{1, 5}, DataType.FLOAT32);

        List<PointF3D> embedding = sample.getEmbedding();
        ByteBuffer buf = ByteBuffer.allocateDirect(99 * 4);

        for (PointF3D landmark : embedding) {
            buf.putFloat(landmark.getX());
            buf.putFloat(landmark.getY());
            buf.putFloat(landmark.getZ());
        }
        buf.rewind();
        for (int i = 1; i <= 99; i++)
            System.out.println(i + ":" + buf.getFloat());

        buf.rewind();
        inputFeature0.loadBuffer(buf);

        try {
            Interpreter interpreter = new Interpreter(loadModelFile(context));
            System.out.println(interpreter);
            interpreter.run(inputFeature0.getBuffer(), output.getBuffer());
            interpreter.close();
        } catch (IOException e) {
            e.printStackTrace();
        }

//            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();
        float[] data = output.getFloatArray();
//            System.out.println("output: " + Arrays.toString(data));
        int i = 0;
        int maxAt = 0;
        for (int j = 0; j < data.length; j++) {
            maxAt = data[j] > data[maxAt] ? j : maxAt;
        }
//        System.out.println("index: " + maxAt);
        for (float val : data) {
            switch (i) {
                case 0:
                    result.putClassConfidence("goddess", val);
                    break;
                case 1:
                    result.putClassConfidence("warrior 2", val);
                    break;
                case 2:
                    result.putClassConfidence("tree", val);
                    break;
                case 3:
                    result.putClassConfidence("plank", val);
                    break;
                case 4:
                    result.putClassConfidence("downward facing dog", val);
                    break;
            }
            i++;
        }
        // Releases model resources if no longer used.
        return getStringFromResult(result);
    }


    public List<String> classify(Pose pose, Context context) {
        ClassificationHistory history = ClassificationHistory.getInstance();
        ClassificationResult result = new ClassificationResult();
        try {
            FinalModel3 dlClassifier = FinalModel3.newInstance(context);
            // Creates inputs for reference.
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 99}, DataType.FLOAT32);
            List<PointF3D> landmarks = extractPoseLandmarks(pose);
//            System.out.println(landmarks);
            if (landmarks.isEmpty()) {
                return getStringFromResult(history.getResult());
            }
            List<PointF3D> embedding = getPoseEmbedding(landmarks);
            ByteBuffer buf = ByteBuffer.allocateDirect(99 * 4);
            boolean justOnce = true;

            for (PointF3D landmark : embedding) {
                if (justOnce) {
//                    System.out.println("landmark X: " + landmark.getX() + " landmark Y: " + landmark.getY() + " landmark Z: " + landmark.getZ());
                }
                buf.putFloat(landmark.getX());
                buf.putFloat(landmark.getY());
                buf.putFloat(landmark.getZ());
            }
            buf.rewind();
//            for (int i = 1; i <= 99; i++)
//                System.out.println("buf: " + buf.getFloat());
//
//            buf.rewind();
            inputFeature0.loadBuffer(buf);

            // Runs model inference and gets result.
            FinalModel3.Outputs outputs = dlClassifier.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();
            float[] data = outputFeature0.getFloatArray();
//            System.out.println("output: " + Arrays.toString(data));
            int i = 0;
            for (float val : data) {
                switch (i) {
                    case 0:
                        result.putClassConfidence("goddess", val);
                        break;
                    case 1:
                        result.putClassConfidence("warrior 2", val);
                        break;
                    case 2:
                        result.putClassConfidence("tree", val);
                        break;
                    case 3:
                        result.putClassConfidence("plank", val);
                        break;
                    case 4:
                        result.putClassConfidence("downward facing dog", val);
                        break;
                }
                i++;
            }
            history.addNewResult(result);
            // Releases model resources if no longer used.
            dlClassifier.close();
            return getStringFromResult(history.getResult());
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }
    }

    public List<String> classifySample(PoseSample sample, Context context) {
        boolean justOnce = true;
        ClassificationResult result = new ClassificationResult();
        try {
            FinalModel3 dlClassifier = FinalModel3.newInstance(context);
            // Creates inputs for reference.
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 99}, DataType.FLOAT32);
            List<PointF3D> landmarks = sample.getLandmarks();
            List<PointF3D> embedding = getPoseEmbedding(landmarks);
            ByteBuffer buf = ByteBuffer.allocateDirect(99 * 4);
            float temp[] = new float[99];
            int q = 0;
            for (PointF3D landmark : embedding) {
                temp[q] = (landmark.getX());
                temp[q+1] = (landmark.getY());
                temp[q+2] = (landmark.getZ());
                
                buf.putFloat(landmark.getX());
                buf.putFloat(landmark.getY());
                buf.putFloat(landmark.getZ());
                q=q+3;
            }
//            System.out.println(sample.getName() + ", " + sample.getClassName() + ", " + Arrays.toString(temp));
//            writeEmbeddingToFile(temp, sample);
            buf.rewind();
//            float y = 1;
//            for (int i = 0; i < 99; i++) {
////                System.out.println(i);
//                y = buf.getFloat();
//                if (temp[i] == y) {
//                    System.out.println("same same! " + y);
//                } else {
//                    System.out.println("different! " + temp + " " + y);
//                }
//
//            }
//            buf.rewind();
            inputFeature0.loadBuffer(buf);

            // Runs model inference and gets result.
            FinalModel3.Outputs outputs = dlClassifier.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();
            float[] data = outputFeature0.getFloatArray();
//            System.out.println("output: " + Arrays.toString(data));
            int i = 0;
            int maxAt = 0;
            for (int j = 0; j < data.length; j++) {
                maxAt = data[j] > data[maxAt] ? j : maxAt;
            }
            System.out.println("should be: " + sample.getClassName());
            System.out.println("found: " + maxAt);
            for (float val : data) {
                switch (i) {
                    case 0:
                        result.putClassConfidence("goddess", val);
                        break;
                    case 1:
                        result.putClassConfidence("warrior 2", val);
                        break;
                    case 2:
                        result.putClassConfidence("tree", val);
                        break;
                    case 3:
                        result.putClassConfidence("plank", val);
                        break;
                    case 4:
                        result.putClassConfidence("downward facing dog", val);
                        break;
                }
                i++;
            }
//            System.out.println(result);
            // Releases model resources if no longer used.
            dlClassifier.close();
            return getStringFromResult(result);
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }
    }

    private void writeEmbeddingToFile(float[] array, PoseSample sample) throws IOException {
        File folder = new File(Environment.getExternalStorageDirectory()
                + "/Folder");
//        File file = new File("Bureaublad/output.csv");
        try (PrintWriter writer = new PrintWriter("embedding.csv")) {
            StringBuilder sb = new StringBuilder();
            sb.append(sample.getName());
            sb.append(',');
            sb.append(sample.getClassName());
            for (float v : array) {
                sb.append(String.valueOf(v));
                sb.append(',');
            }
            sb.append('\n');
            writer.write(sb.toString());
//            System.out.println("done!");

        } catch (FileNotFoundException e) {
//            System.out.println(e.getMessage());
        }

    }
//        File file = new File(CSV_FILE_NAME);
//        FileWriter fw = new FileWriter(file, true);
//        fw.append(sample.getName());
//        fw.append(',');
//        fw.append(sample.getClassName());
//        fw.append(',');
//        for(int i=0;i<array.length;i++)
//        {
//            fw.append(String.valueOf(array[i]));
//            fw.append(',');
//        }
//        fw.append('\n');
//        fw.flush();
//        fw.close();
//    }

    public void justOnce(Context context){
        System.out.println("just once");
        float[] array = {262.0108f,73.56316f,-88.95951f,269.11963f,67.44794f,-83.69705f,272.06412f,67.47657f,-83.70446f,275.0782f,67.44547f,-83.69432f,262.49008f,67.69175f,-68.55418f,260.7513f,67.70081f,-68.53871f,258.99884f,67.79687f,-68.5338f,284.49948f,71.85834f,-49.281384f,263.5199f,71.42814f,18.399015f,269.53873f,83.03383f,-78.049515f,261.9227f,82.76737f,-58.351162f,307.75787f,111.0921f,-44.468323f,250.42915f,114.855545f,57.765083f,296.1561f,161.88007f,-74.911766f,220.71619f,168.93364f,-59.674618f,260.464f,129.03604f,-55.347675f,245.03912f,129.77586f,-211.76167f,249.07013f,122.39861f,-59.645916f,246.10684f,116.17643f,-231.73915f,254.91911f,114.207794f,-51.31389f,253.95793f,110.52682f,-202.16351f,258.36514f,117.90737f,-48.81259f,256.74164f,116.77091f,-203.99918f,314.06018f,193.66504f,-23.974155f,274.28333f,195.7803f,23.950266f,357.34177f,212.4085f,-278.81174f,221.56116f,215.27707f,-38.62177f,360.26434f,295.5976f,-241.86652f,234.08243f,276.7622f,90.8035f,350.42734f,314.31247f,-241.89603f,241.88179f,291.659f,102.497795f,377.01328f,310.214f,-339.65668f,207.25339f,290.0698f,53.394756f};
        float[] array2 = {523.59436f,362.42795f,-369.49542f,536.7346f,347.1613f,-327.58957f,544.13116f,347.15985f,-327.47324f,551.713f,347.02966f,-327.29697f,515.18823f,347.41098f,-319.71365f,507.77298f,347.16254f,-319.63916f,500.35736f,347.1165f,-319.4354f,562.9364f,356.88538f,-125.74426f,492.96713f,356.6258f,-76.4418f,536.5757f,383.45566f,-300.14224f,510.9447f,383.11414f,-285.9455f,601.3229f,439.30127f,16.614452f,455.31555f,443.20242f,29.303854f,744.76245f,457.22552f,-87.061066f,327.48132f,458.49927f,-103.78105f,744.96375f,358.39606f,-269.3017f,330.95898f,342.18747f,-282.7531f,752.1996f,321.4218f,-325.6955f,324.62213f,295.58008f,-345.84766f,732.2085f,316.66837f,-295.4034f,343.83524f,296.56262f,-297.7896f,725.937f,336.36475f,-267.8905f,353.7232f,317.95972f,-274.15555f,589.26685f,642.5726f,-15.399462f,493.631f,641.70715f,16.008888f,746.572f,731.586f,-467.3028f,359.48547f,703.3966f,-454.4346f,749.4755f,875.34937f,-39.395298f,346.64203f,856.63525f,-65.23191f,729.3983f,901.82623f,-4.8170934f,367.19315f,891.66724f,-33.432972f,801.77747f,915.1841f,-185.47577f,280.89783f,885.8697f,-214.46764f};
//        float[] array = {871.00916f, 280.02707f,  -522.30884f,   881.081f,     264.54565f,  -482.8989f,
//                888.2399f,    263.42984f,  -482.99103f,   895.6007f,   262.36108f,  -483.1654f,
//                860.3839f,    268.6582f,   -473.13522f,   853.42456f,   270.06607f,  -472.82462f,
//                846.67523f,   271.78564f,  -472.93167f,   909.1812f,    273.61914f,  -277.68854f,
//                842.46204f,   284.39734f, -240.19626f,   887.31287f,   295.67407f,  -449.51614f,
//                861.8839f,    300.33316f,  -437.01068f,   943.25916f,   347.59998f,  -331.0555f,
//                830.44214f,   354.92136f,  -239.42015f,   959.07715f,   225.55235f,  -563.1047f,
//                797.54486f,  236.4281f,   -363.10138f,   948.84424f,   100.751175f, -714.5804f,
//                793.40753f,   115.30477f,  -451.22018f,   949.08704f,    70.14465f,  -808.9291f,
//                797.10254f,    89.60048f,  -527.5071f,    937.051f,      66.45287f,  -789.73663f,
//                809.3815f,     81.93096f,  -522.1606f,    937.4381f,     79.40931f,  -720.78564f,
//                809.2387f,     90.86572f,  -459.5222f,    920.00836f,   600.26654f,   -43.519745f,
//                840.6041f,    592.6123f,     43.93082f,   891.737f,     815.5092f,   -106.58353f,
//                718.6897f,   703.0979f,   -363.44894f,   875.8328f,    997.8932f,    106.03582f,
//                857.63074f,   726.2907f,    135.99467f,   873.837f,    1017.4097f,    120.73064f,
//                888.2838f,    710.59247f,   191.30246f,   862.3237f,   1069.4464f,    -95.07156f,
//                869.0815f,    778.54205f,   141.03162f };
        List<PointF3D> newList = new ArrayList<>();
        int i = 0;
        while(i<99){
            newList.add(PointF3D.from(array[i], array[i+1], array[i+2]));
            i=i+3;
        }
        List<PointF3D> embedding = getPoseEmbedding(newList);
        System.out.println("yesyesys: " + embedding);
        ClassificationResult result = new ClassificationResult();
        try {
            FinalModel3 dlClassifier = FinalModel3.newInstance(context);
            // Creates inputs for reference.
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 99}, DataType.FLOAT32);

            ByteBuffer buf = ByteBuffer.allocateDirect(99 * 4);
            boolean justOnce = true;

            for (PointF3D landmark : embedding) {
                if (justOnce) {
                    System.out.println("landmark X: " + landmark.getX());
                    justOnce = false;
                }
                buf.putFloat(landmark.getX());
                buf.putFloat(landmark.getY());
                buf.putFloat(landmark.getZ());
            }
            buf.rewind();
            inputFeature0.loadBuffer(buf);

            // Runs model inference and gets result.
            FinalModel3.Outputs outputs = dlClassifier.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();
            float[] data = outputFeature0.getFloatArray();
            System.out.println(Arrays.toString(data));
            i = 0;
            for (float val : data) {
                switch (i) {
                    case 0:
                        result.putClassConfidence("goddess", val);
                        System.out.println("goddess " + val);
                        break;
                    case 1:
                        result.putClassConfidence("warrior 2", val);
                        System.out.println("warrior 2 " + val);
                        break;
                    case 2:
                        result.putClassConfidence("tree", val);
                        System.out.println("tree " + val);
                        break;
                    case 3:
                        result.putClassConfidence("plank", val);
                        System.out.println("plank " + val);
                        break;
                    case 4:
                        result.putClassConfidence("downward facing dog", val);
                        System.out.println("dfd " + val);
                        break;
                }
                i++;
            }
        } catch(IOException e) {
            e.printStackTrace();
        }

    }

    private List<String> getStringFromResult(ClassificationResult classificationResult){
        List<String> listFromResult = new ArrayList<>();
        String conf;
        Set<String> classes = classificationResult.getAllClasses();
        for(String cl : classes) {
            float res = classificationResult.getClassConfidence(cl);
            conf = String.format(
                    Locale.US, "%s : %.2f confidence", cl, res);
            listFromResult.add(conf);
        }
        return listFromResult;
    }

    private static List<PointF3D> extractPoseLandmarks(Pose pose) {
        List<PointF3D> landmarks = new ArrayList<>();
//        List<Float> floatLandmarks = new ArrayList<>();
        for (PoseLandmark poseLandmark : pose.getAllPoseLandmarks()) {
            landmarks.add(poseLandmark.getPosition3D());
//            floatLandmarks.add(poseLandmark.getPosition3D().getX());
//            floatLandmarks.add(poseLandmark.getPosition3D().getY());
//            floatLandmarks.add(poseLandmark.getPosition3D().getZ());
        }
//        System.out.println("sizesss");
//        System.out.println(floatLandmarks.size());
//        System.out.println(landmarks.size());
        return landmarks;
    }
}
