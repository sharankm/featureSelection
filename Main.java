package com.ai;

import java.io.*;
import java.util.*;

public class Main {

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        System.out.println("Welcome to Sharan's Feature Selection Algorithm.\nType in the name of the file to test: ");
        File dataFile = new File("cs_205_large48.txt");
        System.out.println("\nType the number of the algorithm you want to run.\n" +
                "1.\tForward Selection\n" +
                "2.\tBackward Elimination.\n" +
                "3.\tSpecial Algorithm.\n");
        Integer algoNumber = sc.nextInt();
        Double minimum = 0.0;
        Double maximum = 0.0;
        Integer featureCount = 0;
        Integer rowCount = 0;
        Map<RowInstance, Integer> verifyMap = new LinkedHashMap<>();
        if (!dataFile.exists()) {
            System.out.println("Mentioned File doesn't exist!");
            System.exit(1);
        }
        try {
            FileReader freader = new FileReader(dataFile);
            BufferedReader br = new BufferedReader(freader);
            String line;
            while ((line = br.readLine()) != null) {
                String[] strings = line.split("\\s+");
                featureCount = strings.length - 2;
                rowCount++;
                for (int i = 1; i < strings.length; i++) {
                    if (Double.parseDouble(strings[i]) < minimum)
                        minimum = Double.parseDouble(strings[i]);
                    if (Double.parseDouble(strings[i]) > maximum)
                        maximum = Double.parseDouble(strings[i]);
                }
            }
            System.out.println("This dataset has " + featureCount + " features (not including the class attribute), with " + rowCount + " instances.");
            br.close();
            System.out.print("\nPlease wait while I normalize the data...");
            freader = new FileReader(dataFile);
            br = new BufferedReader(freader);
            while ((line = br.readLine()) != null) {
                String[] strings = line.split("\\s+");
                Double classNumber = Double.parseDouble(strings[1]);
                RowInstance row = new RowInstance();
                for (int feature = 2; feature < strings.length; feature++) {
                    Double normalizedVal = (Double.parseDouble(strings[feature]) - minimum) / (maximum - minimum);
                    row.getFeatureList().add(normalizedVal);
                    verifyMap.put(row, classNumber.intValue());
                }
            }
            br.close();
            System.out.println("  Done!\n");
            System.out.println("Beginning Search.\n");
            Set<Integer> bestFeatures = new HashSet<>();
            Set<Integer> finalFeatures = new HashSet<>();
            Double highestAccuracy = 0.0;
            long startTime = System.nanoTime();
            if (algoNumber == 1) {
                highestAccuracy = forwardSelection(featureCount, bestFeatures, finalFeatures, verifyMap, highestAccuracy);
            } else if (algoNumber == 2) {
                highestAccuracy = backwardSelection(featureCount, bestFeatures, finalFeatures, verifyMap, highestAccuracy);
            } else if (algoNumber == 3) {
                highestAccuracy = specialAlgorithm(featureCount, bestFeatures, finalFeatures, verifyMap, highestAccuracy);
            }
            long endTime = System.nanoTime();
            System.out.println("\nFinished search!! The best feature subset is " + finalFeatures.toString() + " which has an accuracy of " + highestAccuracy * 100 + "%\n");
            System.out.println("TIme Elapsed(milliseconds): " + (endTime - startTime)/1000000 );
        } catch (Exception e) {
            System.out.println("Issue reading the file" + e);
            System.exit(1);
        }
    }

    private static Double getAccuracy(Map<RowInstance, Integer> verifyMap, Integer featureCount, Set<Integer> searchFeatures) {
        int trueCount = 0;
        for (RowInstance row1 : verifyMap.keySet()) {
            RowInstance closestRow = null;
            Double minDistance = 2.0;
            for (RowInstance row2 : verifyMap.keySet()) {
                if (row1 != row2) {
                    Double distance = 0.0;
                    List<Double> list1 = row1.getFeatureList();
                    List<Double> list2 = row2.getFeatureList();
                    for (int i = 0; i < featureCount; i++) {
                        if (searchFeatures.contains(i + 1))
                            distance = distance + Math.pow(list2.get(i) - list1.get(i), 2);
                    }
                    distance = Math.sqrt(distance);
                    if (distance < minDistance) {
                        minDistance = distance;
                        closestRow = row2;
                    }
                }
            }
            if (verifyMap.get(row1).equals(verifyMap.get(closestRow)))
                trueCount++;
        }
        Double accuracy = (double) trueCount / verifyMap.keySet().size();
        System.out.println("\tUsing feature(s) " + searchFeatures.toString() + " accuracy is " + accuracy * 100 + "%");
        return accuracy;
    }

    private static Double forwardSelection(Integer featureCount, Set<Integer> bestFeatures, Set<Integer> finalFeatures, Map<RowInstance, Integer> verifyMap, Double highestAccuracy) {
        for (int i = 1; i <= featureCount; i++) {
            Double bestInnerAccuracy = 0.0;
            Integer bestInner = 0;
            for (int j = 1; j <= featureCount; j++) {
                if (!bestFeatures.contains(j)) {
                    Set<Integer> searchFeatures = new HashSet<>();
                    searchFeatures.add(j);
                    searchFeatures.addAll(bestFeatures);
                    Double accuracy = getAccuracy(verifyMap, featureCount, searchFeatures);
                    if (accuracy > highestAccuracy) {
                        highestAccuracy = accuracy;
                        bestInner = j;
                    }
                    if (accuracy > bestInnerAccuracy) {
                        bestInnerAccuracy = accuracy;
                        bestInner = j;
                    }
                }
            }
            if (bestInnerAccuracy.equals(highestAccuracy)) {
                bestFeatures.add(bestInner);
                finalFeatures.clear();
                finalFeatures.addAll(bestFeatures);
                System.out.println("\nFeature set " + bestFeatures.toString() + " was best, accuracy is " + highestAccuracy * 100 + "%\n");
            } else {
                bestFeatures.add(bestInner);
                System.out.println("\n(Warning, Accuracy has decreased! Continuing search in case of local maxima)");
                System.out.println("Feature set " + bestFeatures.toString() + " was best, accuracy is " + bestInnerAccuracy * 100 + "%\n");
            }
        }
        return highestAccuracy;
    }

    private static Double backwardSelection(Integer featureCount, Set<Integer> bestFeatures, Set<Integer> finalFeatures, Map<RowInstance, Integer> verifyMap, Double highestAccuracy) {
        for (int i = 1; i <= featureCount; i++) {
            bestFeatures.add(i);
            finalFeatures.add(i);
        }
        System.out.println("\nFeature set " + bestFeatures.toString() + " was best, accuracy is " + getAccuracy(verifyMap, featureCount, bestFeatures) * 100 + "%\n");
        for (int i = 1; i < featureCount; i++) {
            Double bestInnerAccuracy = 0.0;
            Integer worstInner = 0;
            for (int j = featureCount; j > 0; j--) {
                if (bestFeatures.contains(j)) {
                    Set<Integer> searchFeatures = new HashSet<>();
                    searchFeatures.addAll(bestFeatures);
                    searchFeatures.remove(j);
                    Double accuracy = getAccuracy(verifyMap, featureCount, searchFeatures);
                    if (accuracy > highestAccuracy) {
                        highestAccuracy = accuracy;
                        worstInner = j;
                    }
                    if (accuracy > bestInnerAccuracy) {
                        bestInnerAccuracy = accuracy;
                        worstInner = j;
                    }
                }
            }
            if (bestInnerAccuracy.equals(highestAccuracy)) {
                bestFeatures.remove(worstInner);
                finalFeatures.clear();
                finalFeatures.addAll(bestFeatures);
                System.out.println("\nFeature set " + bestFeatures.toString() + " was best, accuracy is " + highestAccuracy * 100 + "%\n");
            } else {
                bestFeatures.remove(worstInner);
                System.out.println("\n(Warning, Accuracy has decreased! Continuing search in case of local maxima)");
                System.out.println("Feature set " + bestFeatures.toString() + " was best, accuracy is " + bestInnerAccuracy * 100 + "%\n");
            }
        }
        return highestAccuracy;
    }

    private static Double specialAlgorithm(Integer featureCount, Set<Integer> bestFeatures, Set<Integer> finalFeatures, Map<RowInstance, Integer> verifyMap, Double highestAccuracy) {
        for (int i = 1; i <= featureCount; i++) {
            bestFeatures.add(i);
            finalFeatures.add(i);
        }
        System.out.println("\nFeature set " + bestFeatures.toString() + " was best, accuracy is " + getAccuracy(verifyMap, featureCount, bestFeatures) * 100 + "%\n");
        while (bestFeatures.size() > 1) {
            Double bestInnerAccuracy = 0.0;
            List<Double> accList = new ArrayList<>();
            Map<Integer, Double> accMap = new LinkedHashMap<>();
            for (int j = 1; j <= featureCount; j++) {
                if (bestFeatures.contains(j)) {
                    Set<Integer> searchFeatures = new HashSet<>();
                    searchFeatures.addAll(bestFeatures);
                    searchFeatures.remove(j);
                    Double accuracy = getAccuracy(verifyMap, featureCount, searchFeatures);
                    if (accuracy > highestAccuracy) {
                        highestAccuracy = accuracy;
                    }
                    if (accuracy > bestInnerAccuracy) {
                        bestInnerAccuracy = accuracy;
                    }
                    accList.add(accuracy);
                    accMap.put(j, accuracy);
                }
            }
            Collections.sort(accList);
            Double worst1 = accList.get(accList.size() - 1);
            Double worst2 = accList.get(accList.size() - 2);
            if(worst1.equals(worst2)){
                int found = 0;
                for(Integer key: accMap.keySet()){
                    if(accMap.get(key).equals(worst1)){
                        bestFeatures.remove(key);
                        found++;
                        if(found == 2)
                            break;
                    }
                }
            } else{
                for(Integer key: accMap.keySet()){
                    if(accMap.get(key).equals(worst1))
                        bestFeatures.remove(key);
                    if(accMap.get(key).equals(worst2))
                        bestFeatures.remove(key);
                }
            }
            if (bestInnerAccuracy.equals(highestAccuracy)) {
                finalFeatures.clear();
                finalFeatures.addAll(bestFeatures);
                System.out.println("\nFeature set " + bestFeatures.toString() + " was best, accuracy is " + highestAccuracy * 100 + "%\n");
            } else {
                System.out.println("\n(Warning, Accuracy has decreased! Continuing search in case of local maxima)");
                System.out.println("Feature set " + bestFeatures.toString() + " was best, accuracy is " + bestInnerAccuracy * 100 + "%\n");
            }
        }
        return highestAccuracy;
    }
}
