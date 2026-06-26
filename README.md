# ModePerbandinganDataset
Ini menggunakan algoritma BAB 3 namun menggunakan 3 open dataset yang ada dan membandingkannya.

=============================================================
  DTS-FA: Dynamic Trust Score dengan Federated Aggregation
  Deteksi Serangan DDoS pada SDN Multi-Controller

  Implementasi: Disertasi Bab III — Algoritma 1 (DTS-FA)

  Dataset:
    1. CICIDS2017  → sweety18/cicids2017-full-dataset
    2. CICDDoS2019 → kristianfrossos/cicddos2019
    3. DDoS-SDN    → aikenkazin/ddos-sdn-dataset

  Output (file terpisah):
    loss_vs_round_<dataset>.png    — per dataset (2 panel)
    confusion_matrix_<dataset>.png — per dataset
    accuracy_vs_dataset.png
    metrics_comparison.pngrikieHarByHanAi/ModePerbandinganDataset
    results_summary.csv

  Setup:
   
    pip install kaggle torch scikit-learn seaborn pandas matplotlib psutil
    kaggle.json sudah ada di /home/rikie/.kaggle/kaggle.json
=============================================================
