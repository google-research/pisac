# Predictive Information Accelerates Learning in RL

[Kuang-Huei Lee][leekh], [Ian Fischer][iansf], [Anthony Liu][aliu],
[Yijie Guo][yguo], [Honglak Lee][honglak], [John Canny][canny],
[Sergio Guadarrama][sguada]

NeurIPS 2020

![cheetah_video](https://user-images.githubusercontent.com/4847452/95011238-33857a00-05e4-11eb-9224-7913a8859381.gif)
![walker_video](https://user-images.githubusercontent.com/4847452/95011273-50ba4880-05e4-11eb-87d2-8a5c0ab54bc7.gif)
![bic_video](https://user-images.githubusercontent.com/4847452/95011243-3c764b80-05e4-11eb-907e-3e0790bff4e1.gif)
![cartpole_video](https://user-images.githubusercontent.com/4847452/95011256-413aff80-05e4-11eb-964a-37a333412245.gif)
![finger_video](https://user-images.githubusercontent.com/4847452/95011270-4d26c180-05e4-11eb-9524-0db5dbc7c7ce.gif)

This repository hosts the open source implementation of PI-SAC, the
reinforcement learning agent introduced in
[Predictive Information Accelerates Learning in RL][paper]. PI-SAC combines the
Soft Actor-Critic Agent with an additional objective that learns compressive
representations of predictive information. PI-SAC agents can substantially
improve sample efficiency and returns over challenging baselines on tasks from
the [DeepMind Control Suite][dmc_paper] of vision-based continuous control
environments, where observations are pixels.

[paper]: https://arxiv.org/abs/2007.12401
[pdf_paper]: https://arxiv.org/pdf/2007.12401.pdf
[leekh]: https://scholar.google.com/citations?user=rE7-N30AAAAJ
[iansf]: https://scholar.google.com/citations?user=Z63Zf_0AAAAJ
[aliu]: https://scholar.google.com/citations?user=TjEqCOAAAAAJ
[yguo]: https://scholar.google.com/citations?user=ONuIPv0AAAAJ
[honglak]: https://scholar.google.com/citations?user=fmSHtE8AAAAJ
[canny]: https://scholar.google.com/citations?user=LAv0HTEAAAAJ
[sguada]: https://scholar.google.com/citations?user=gYiCq88AAAAJ
[dmc_paper]: https://arxiv.org/abs/1801.00690

If you find this useful for your research, please use the following to
reference:

```
@article{lee2020predictive,
  title={Predictive Information Accelerates Learning in RL},
  author={Lee, Kuang-Huei and Fischer, Ian and Liu, Anthony and Guo, Yijie and Lee, Honglak and Canny, John and Guadarrama, Sergio},
  journal={arXiv preprint arXiv:2007.12401},
  year={2020}
}
```

## Methods

![pi2small](https://user-images.githubusercontent.com/4847452/95029558-e7771b80-065d-11eb-8f8b-7c2ecffc1222.png)

PI-SAC learns compact representations of the predictive information
I(X_past;Y_future) that captures the environment transition dynamics, in
addition to actor and critic learning. We capture the predictive information in
a representation Z by maximizing I(Y_future;Z) and minimizing
I(X_past;Z|Y_future) to compress out the non-predicitve part for better
generalization, which reflects in better sampled efficiency, returns, and
transferability. When interacting with the environment, it simply executes the
actor model.

Find out more:

-   [PDF paper][pdf_paper]

## Training and Evaluation

To train the model(s) in the paper with periodic evaluation, run this command:

```train
python -m pisac.run --root_dir=/tmp/pisac_cartpole_swingup \
--gin_file=pisac/config/pisac.gin \
--gin_bindings=train_pisac.train_eval.domain_name=\'cartpole\' \
--gin_bindings=train_pisac.train_eval.task_name=\'swingup\' \
--gin_bindings=train_pisac.train_eval.action_repeat=4 \
--gin_bindings=train_pisac.train_eval.initial_collect_steps=1000 \
--gin_bindings=train_pisac.train_eval.initial_feature_step=5000
```

We use `gin` to config hyperparameters. The default configs are specificed in
`pisac/config/pisac.gin`. To reproduce the main DM-Control experiments, you need
to specify different `domain_name`, `task_name`, `action_repeat`,
`initial_collect_steps`, `initial_feature_step` for each environment.

`domain_name` | `task_name`    | `action_repeat` | `initial_collect_steps` | `initial_feature_step`
:------------ | :------------- | :-------------- | :---------------------- | :---------------------
cartpole      | swingup        | 4               | 1000                    | 5000
cartpole      | balance_sparse | 2               | 1000                    | 5000
reacher       | easy           | 4               | 1000                    | 5000
ball_in_cup   | catch          | 4               | 1000                    | 5000
finger        | spin           | 1               | 10000                   | 0
cheetah       | run            | 4               | 10000                   | 10000
walker        | walk           | 2               | 10000                   | 10000
walker        | stand          | 2               | 10000                   | 10000
hopper        | stand          | 2               | 10000                   | 10000

To use multiple gradient steps per environment step, change
`train_pisac.train_eval.collect_every` to a number larger than 1.

## Results

### DeepMind Control Suite

![pisac_full2](https://user-images.githubusercontent.com/4847452/95033853-71ca7a00-0674-11eb-8a0f-afea8e63b4bf.png)

\*gs: number of gradient steps per environment step

## Requirements

The PI-SAC code uses Python 3 and these packages:

-   tensorflow-gpu==2.3.0
-   tf_agents==0.6.0
-   tensorflow_probability
-   dm_control (`egl` [rendering option][rendering] recommended)
-   gym
-   imageio
-   matplotlib
-   scikit-image
-   scipy
-   gin
-   pstar
-   qj

If you ever see that dm_control complains about some threading issues, please
try adding `--gin_bindings=train_pisac.train_eval.drivers_in_graph=False` to put
dm_control environment outside of the TensorFlow graph.

[rendering]: https://github.com/deepmind/dm_control#rendering

Disclaimer: This is not an official Google product.
