# ============================================
# Short-test version of ContentAwareEnvV2
# ============================================

env = ContentAwareEnvV2(
    use_real_traces=True,
    total_chunks=10  # فقط 10 chunk برای تست سریع
)

# کاهش شدت penalty ها برای تست
env.reward_func.rebuffer_penalty = 1.0
env.reward_func.smoothness_penalty = 0.2

# محدود کردن max_download_time برای safety
env.step = lambda action, original_step=env.step: original_step(action)  # از step اصلی استفاده می‌کنیم

print("Running short test episode...")

state = env.reset(video_id=1, split='train')
total_reward = 0
total_rebuffer = 0

for i in range(env.total_chunks):
    # انتخاب action ساده: همیشه bitrate متوسط (index 2)
    action = 2
    next_state, reward, done, info = env.step(action)
    
    total_reward += reward
    total_rebuffer += info['rebuffer_time']
    
    print(f"Step {i}: action={action} ({env.bitrate_levels[action]} kbps), "
          f"reward={reward:+.3f}, buffer={info['buffer']:.1f}s, "
          f"rebuffer={info['rebuffer_time']:.2f}s, throughput={info['throughput']:.0f}kbps")
    
    if done:
        break

print(f"\nTest episode finished. Total reward: {total_reward:.3f}, Total rebuffer: {total_rebuffer:.2f}s")
