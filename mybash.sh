echo "Testing Pensieve Reward..."
python models/pensieve_reward.py

# ==========================================
# Step 2: Check که همه فایل‌ها هستن
# ==========================================
echo ""
echo "Checking files..."
ls -lh models/content_aware_model.py
ls -lh models/content_aware_env_v2.py
ls -lh models/trace_loader.py
ls -lh models/pensieve_reward.py
ls -lh models/ppo_trainer.py
ls -lh scripts/training/train_ppo.py

# ==========================================
# Step 3: START TRAINING! 🚀
# ==========================================
echo ""
echo "Starting PPO Training..."
echo ""
python scripts/training/train_ppo.py
