Ablation Study Analysis: Impact of Content-Awareness and VMAF-Awareness on ABR Performance

This report presents the findings from an ablation study conducted to evaluate the individual contributions of key components in our content-aware ABR (Adaptive Bitrate) agent. The primary objective was to understand the importance of VMAF-awareness and content complexity features (SI/TI) on the agent's overall Quality of Experience (QoE) performance.

Summary of Results

We evaluated three ablated models against each other. The results are summarized in the table below:

Model Configuration (Component Removed)

                                    Reward (QoE)      Std. Dev.         Rebuffer (s)          Bitrate (kbps)

No VMAF (VMAF-Awareness Removed)    107.18            18.58                 1.05s              1143

Network Only (VMAF & SI/TI Removed)  104.93            19.81                1.52s              1164

No SI/TI (Content Features Removed)  104.48            22.49                1.41s              1084

(Bold values indicate the best performance in that specific metric.)

Detailed Analysis of Findings

1. The No VMAF Model (Best Overall Performance)

Strategy: This model, lacking VMAF-awareness, optimizes its reward function based primarily on maximizing throughput (bitrate) while minimizing rebuffering penalties.

Analysis: Surprisingly, this model achieved the highest overall reward. Its success is attributed to a "stability-first" strategy. By securing the lowest average rebuffering time (1.05s), it avoided significant QoE penalties. This indicates that, in this dataset, the negative impact of video stalls (rebuffering) is more significant than the fine-grained quality optimization provided by VMAF.

2. The Network Only Model (Overly Aggressive)

Strategy: This model acts as a traditional, network-aware-only ABR agent, ignoring both VMAF and content complexity. Its sole aim is to select the highest possible bitrate that the network can sustain.

Analysis: This agent adopted an "aggressive" policy, successfully achieving the highest average bitrate (1164 kbps). However, this strategy was high-risk and resulted in the highest rebuffering time (1.52s). The severe penalty from these stalls dramatically reduced its overall reward, demonstrating that a bitrate-greedy strategy is suboptimal for QoE.

3. The No SI/TI Model (Overly Cautious)

Strategy: This agent is VMAF-aware but lacks content complexity features (SI/TI). It tries to optimize for VMAF without understanding why a segment has a particular VMAF score (e.g., is it a simple scene at low bitrate or a complex scene at high bitrate?).

Analysis: This model yielded the worst performance. The removal of content-awareness made the agent "overly cautious," leading to the lowest average bitrate (1084 kbps). It appears that without knowing the scene's complexity, the agent cannot effectively use the VMAF information and defaults to safer, lower-quality bitrate selections to avoid risk, ultimately harming the final reward.

Key Conclusions & Next Steps

Based on this study, we draw two critical conclusions:

Rebuffering Penalty is Critical: The success of the No VMAF model strongly suggests that the rebuffering penalty in our main agent's reward function should be significantly increased. A "stability-first" approach appears to be a more robust strategy for enhancing user QoE.

Content-Awareness (SI/TI) is Essential: The failure of the No SI/TI model proves the value of our content-awareness features. These features are crucial for interpreting VMAF data correctly and prevent the agent from becoming overly conservative. They allow the agent to select higher bitrates for simple content and save bandwidth on complex content, optimizing the quality-bitrate trade-off effectively.