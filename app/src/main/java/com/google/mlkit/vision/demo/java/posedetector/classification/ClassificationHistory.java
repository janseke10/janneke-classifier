package com.google.mlkit.vision.demo.java.posedetector.classification;

import static java.lang.Double.isNaN;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class ClassificationHistory {
    List<ClassificationResult> history;

    private static ClassificationHistory instance = null;

    private ClassificationHistory() {
        history = new ArrayList<>();
    }

    public static ClassificationHistory getInstance() {
        if (instance == null)
            instance = new ClassificationHistory();
        return instance;
    }

    public void addNewResult(ClassificationResult result){
        if(!isNaN(result.getClassConfidence("tree"))&&history.size()>=20){
            history.remove(0);
            history.add(result);
        }
        if(!isNaN(result.getClassConfidence("tree")) && history.size()<20){
            history.add(result);
        }
    }

    public ClassificationResult getResult(){
        ClassificationResult averageResult = new ClassificationResult();
        Map<String, Float> avResult = new HashMap<>();
        for(ClassificationResult result : history){
            Set<String> classes = result.getAllClasses();
            for(String cl : classes){
                if(avResult.containsKey(cl)&& avResult.get(cl)!=null){
                    avResult.put(cl, avResult.get(cl)+result.getClassConfidence(cl));
                } else {
                    avResult.put(cl, result.getClassConfidence(cl));
                }
            }
            for (Map.Entry<String, Float> entry : avResult.entrySet()) {
                int size = history.size();
                String key = entry.getKey();
                Float value = entry.getValue();
                averageResult.putClassConfidence(key, value/size);
            }
        }
        return averageResult;
    }
}
