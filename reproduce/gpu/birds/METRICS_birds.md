# MLSP-2013-Birds — MLE-bench result

## Result (official grader, single graded submission)
- Official AUC (pooled ROC-AUC over all Id rows): 0.93143
- Medal: SILVER  (bronze 0.87372 / silver 0.90038 / gold 0.93527 / median 0.86572)
- valid_submission: True ; silver_medal: True ; any_medal: True
- Held-out OOF pooled AUC (5-fold within 258 train recordings): 0.90512

## Metric
grade.py = sklearn.roc_auc_score over the FLAT (Id, Probability) long-format table
(all 64 test recordings x 19 species pooled = 1216 rows). NOT column-wise mean.
=> cross-species probability calibration matters; rank-averaging blends work well.

## Data understanding
- 322 recordings (10s, 16kHz mono WAV). CVfolds_2.txt: fold 0 = 258 train, fold 1 = 64 test.
- Labels: rec_labels_test_hidden.txt (train rows list species ids; "?" = test).
- species_list.txt: 19 species (class_id 0..18).
- Submission: Id = rec_id*100 + species_id, Probability. 1216 rows, must match sample_submission Id order.

## Model (winning blend = w0.6_cnn, selected by OOF ONLY, graded once)
Final = 0.6 * rank(CNN) + 0.4 * rank(classical_blend), then min-max scaled to [0,1]
(scaling is AUC-invariant; needed so values are valid probabilities in [0,1]).

1. CNN (main lever): resnet18 pretrained (ImageNet), fc->19-way, BCEWithLogitsLoss
   (pos_weight for class imbalance). 5-fold CV, best-epoch by fold-internal val AUC,
   3x TTA. CNN-only OOF pooled AUC = 0.88837.
   - Spectrogram: log-mel, n_fft=1024, hop=256, 128 mels, fmin200/fmax8000.
     Per-frequency-bin BACKGROUND SUBTRACTION (subtract per-mel median over time,
     clip>=0) = the whale-proven denoising lever. Per-image z-norm, resize 224x224,
     3-channel, ImageNet norm.
   - Aug: SpecAugment (2 freq + 2 time masks), random time-crop (wrap-pad), mixup(0.4).
   - 30 epochs, AdamW lr 8e-4 wd 1e-3, cosine. CPU (no GPU on VM).
2. classical_blend = 0.5*rank(lgb on classical feats) + 0.5*rank(logreg on [classical+resnet50 emb]).
   Classical feats: histogram_of_segments (100-d, provided), aggregated segment_features
   (provided), my log-spectrogram band stats w/ bg-subtraction. resnet50 frozen embedding (2048-d).
   lgb_classical OOF=0.84583 ; lr_all OOF=0.82534 (both < CNN, add diversity in blend).

## Selection honesty
- Blend weight w=0.6 chosen by held-out OOF pooled AUC (candidates 0.4..0.8), NOT by test.
- Test set graded exactly once for the OOF-selected model (0.93143). No test-tuning.
- (An abandoned attempt to add effb0/resnet34 backbones for a gold push was killed;
  those saved no artifacts and never touched the submission.)

## Files
- birds_submission.csv  = final graded submission (silver)
- birds_submission_silver.csv = identical backup
- verify.py = format check + single official grade
- birds.py / embed.py / finetune.py / final.py = pipeline (features / resnet50 emb / CNN finetune / blend+grade)
- birds_cnn.npy (resnet18 OOF+test), birds_emb.npy (resnet50 embeddings)
