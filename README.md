
This repo contains source code for the natural signal Atari environments, introduced in
the paper [Natural Environment Benchmarks for Reinforcement Learning](https://arxiv.org/abs/1811.06032).

<div align="center">
  <img src="demo.gif" width="700px" />
</div>

## Instructions

1. Install dependencies with `pip install gym[atari] pygame scikit-video opencv-python`
2. Prepare a directory of images or videos
3. Play with new versions of Atari games with the following commands:

```
# Inject gaussian noise to the observations
./natural_env.py --env BreakoutNoFrameskip-v4 --imgsource noise

# Inject some video signals to the observations
./natural_env.py --env SpaceInvadersNoFrameskip-v4 --imgsource videos --resource-files "~/my/videos/*.mp4"
```

## License
This project is under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE) for details.
