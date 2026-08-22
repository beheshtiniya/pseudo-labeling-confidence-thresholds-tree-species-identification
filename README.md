

## 📂 Execution Instructions (English)

1. **Download the images** from [yun.ir/q24k4b]([https://yun.ir/q24k4b](https://zenodo.org/records/21385214)) and place them inside the `images/` folder.
2. Place the files `train_labels.csv`, `test_labels.csv`, and `val_labels.csv` from [yun.ir/q24k4b]([https://yun.ir/q24k4b](https://zenodo.org/records/21385214)) in the same directory as the `images/` folder.  
3. Run the script `pseudo_labeling_search_best_threshold_final.py`.  
   ⚠️ **Before running**, update all file paths to match your local directory.  
   This script executes 4 self-training repetitions for each threshold.
4. Run final_test.py for each repetition to generate predictions using the trained model, filter the predictions based on IoU, and compute the confusion matrix.
5. Then, run map.py for each repetition to compute the final mAP score.
6. Copy the resulting confusion matrices into the appropriate section of the Excel file. Explanations are available in [`pseudo_labeling_results_explain.md`](pseudo_labeling_results_explain.md).
Metrics will be automatically calculated. Compare results across thresholds to identify the optimal one.

---
