Installation

metaworld
- https://github.com/rlworkgroup/metaworld
- https://github.com/openai/mujoco-py/issues/268
- https://github.com/openai/mujoco-py/issues/647


Training Expert Sample Command
`python expert_train --env-index 0 --n-task 10`

Environment Index to Names
0. reach-v2
1. push-v2
2. pick-place-v2
3. door-open-v2
4. drawer-open-v2
5. drawer-close-v2
6. button-press-topdown-v2
7. peg-insert-side-v2
8. window-open-v2
9. window-close-v2

Todo
1. Verify a single feedforward network can perform behavioural cloning on a single task
2. Verify a single feedforward network can perform behavioural cloning on multiple tasks when trained simultaneously
3. Verify a single feedforward network goes through catastrophic forgetting on multiple tasks when trained sequentially
4. Verify a single feedforward network does not catastrophically forget when trained sequentially with perfect replay
5. verify a single feedforward network does not catastrophically forget when trained sequentially with generative replay
6. Verify a single feedforward network does not catastrophically forget when trained sequentially with internal generative replay with perfect pre-trained embeddings
7. Verify a single feedforward network does not catastrophically forget when trained sequentially with internal generative replay with no pre-trained embeddings
