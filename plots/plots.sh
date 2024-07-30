# figure 3
python scripts/plot_metric.py --wandb_group ours lwf si ewc --name "DCL" "LwF" "SI" "EWC" --metric_name "test/accuracy" --out_path comparison_regularization.pdf --max_steps 500 --wandb_entity "codis/disco" --grid --legend_loc right

# figure 4
python scripts/plot_metric.py --wandb_group ours replay_1k replay_2k replay_5k replay_10k replay_20k --name DCL "Replay (1k)" "Replay (2k)" "Replay (5k)" "Replay (10k)" "Replay (20k)" --metric_name "test/accuracy" --out_path plots/comparison_replay.pdf --max_steps 500 --grid --legend_loc right

# figure 5
python scripts/plot_metric.py --wandb_group ours l2p --name DCL "L2P" --metric_name "test/accuracy" --out_path plots/comparison_pretrained.pdf --max_steps 100 --legend_loc right --grid --legend_locright

# figure 6
python scripts/plot_gpt_benchmark.py

# figure 7
python scripts/plot_metric.py --wandb_group ours --name "DCL" --metric_name "test/accuracy" --out_path plots/comparison_contrastive.pdf --max_steps 200 --grid --legend_loc right --include_contrastive

# figure 8
python scripts/plot_precision_recall.py

# figure 1 (appendix)
python scripts/plot_metric.py --wandb_group ours agem --name DCL "A-GEM" --metric_name "test/accuracy" --out_path comparison_agem.pdf --max_steps 200 --legend_loc right --grid

# figure 2 (appendix)
python scripts/plot_metric.py --wandb_group ours der --name DCL "DER" --metric_name "test/accuracy" --out_path comparison_der.pdf --max_steps 200 --fig_width 4 --fig_height 3 --grid --legend_loc right --force_download

# figure 3 (appendix)
python scripts/plot_metric.py --wandb_group  ours_online ours_3 ours --name "DCL online" "DCL 3 epochs" "DCL 5 epochs" --metric_name "test/accuracy" --out_path comparison_ours.pdf --max_steps 500 --grid --legend_loc right
