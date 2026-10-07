# Third-party source and attribution

The project-specific preprocessing, baselines, experiment runner, synthetic
generator, tests, and documentation are covered by the root MIT license.
That license does not replace the licenses of adapted upstream components.

## Graph WaveNet

`models/graph_wavenet.py` contains adapted Graph WaveNet components. The names,
adjacency utilities, graph convolutions, adaptive adjacency initialization,
configuration interface, and dynamic-layer formula closely match
[LibCity's GWNET implementation](https://github.com/LibCity/Bigscity-LibCity/blob/master/libcity/model/traffic_speed_prediction/GWNET.py).
The historical local source explicitly credits LibCity for the layer formula.
This establishes an upstream relationship by source comparison; the original
upstream commit and exact intermediary copy were not recorded in Git history.
The cited sources were reviewed on 2026-10-07; this is a review date, not an
invented original adaptation date or source revision.

The LibCity-derived module is distributed under Apache-2.0, with the complete
license retained in `third_party/licenses/LibCity-Apache-2.0.txt`.
The original [Graph WaveNet implementation](https://github.com/nnzhan/Graph-WaveNet)
by Zonghan Wu is MIT-licensed; its original copyright and permission notice
are retained in `third_party/licenses/Graph-WaveNet-MIT.txt`.

Local modifications include sequence-shaped multivariate inputs/outputs,
separate input and target scaling, handover-graph cell ordering, Conv2d tensor
handling, training/validation integration, and checkpoint restoration of
scalers and graph state. Graph WaveNet is an existing method; this project
applies and adapts it to telecom KPI forecasting rather than claiming to have
introduced the architecture.

## References

Wu, Z., Pan, S., Long, G., Jiang, J., and Zhang, C. (2019).
*Graph WaveNet for Deep Spatial-Temporal Graph Modeling*. IJCAI 2019.
[Paper](https://arxiv.org/abs/1906.00121).

Wang, J., Jiang, J., Jiang, W., Li, C., and Zhao, W. X. (2021).
*LibCity: An Open Library for Traffic Prediction*. ACM SIGSPATIAL 2021.
[Project](https://github.com/LibCity/Bigscity-LibCity),
[license](https://github.com/LibCity/Bigscity-LibCity/blob/master/LICENSE.txt).
