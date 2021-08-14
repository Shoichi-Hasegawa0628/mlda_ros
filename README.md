# 'mlda_ros' Package

The `mlda_ros` enables the object categorization using MLDA.

*   Maintainer: Shoichi Hasegawa ([hasegawa.shoichi@em.ci.ritsumei.ac.jp](mailto:hasegawa.shoichi@em.ci.ritsumei.ac.jp)).
*   Author: Shoichi Hasegawa ([hasegawa.shoichi@em.ci.ritsumei.ac.jp](mailto:hasegawa.shoichi@em.ci.ritsumei.ac.jp)).

**Content:**

*   [Launch](#launch)
*   [Files](#files)
*   [References](#references)

## Launch

*   `em_mlda_main.launch`: The whole system of the MLDA.

## Files

*   `__init__.py`: Initialization of topics and parameters.
*   `em_mlda_learn.py`: Learn the object categorization.
*   `em_mlda_data_image.py`: Make the BoF from images.
*   `em_mlda_data_word.py`: Make the BoW from words.
*   `em_mlda_data_joint_load.py`: Make the BoF from joint load.
*   `em_mlda_data_human_input.py`: Get the signal from the human input with HoloLens.
*   `em_mlda_hololens_coordinate.py`: Visualize the ease of categorization of objects.
*   `em_mlda_hololens_category_sample.py`: Visualize the sample objects for each category.

## References
*   Implementions of LDAs: [https://github.com/is0383kk/LDAs](https://github.com/is0383kk/LDAs)
*   Multimodal Object Dataset 165: [https://hp.naka-lab.org/subpages/mod165.html](https://hp.naka-lab.org/subpages/mod165.html)
