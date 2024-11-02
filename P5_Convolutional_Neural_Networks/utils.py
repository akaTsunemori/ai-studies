import matplotlib.pyplot as plt
import numpy as np

plt.rcdefaults()
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle

NumDots = 4
NumConvMax = 24
NumFcMax = 10
White = 1.0
Light = 0.7
Medium = 0.5
Dark = 0.3
Darker = 0.15
Black = 0.0


def add_layer(
    patches,
    colors,
    size=(24, 24),
    num=5,
    top_left=[0, 0],
    loc_diff=[10, -10],
):
    # add a rectangle
    top_left = np.array(top_left)
    loc_diff = np.array(loc_diff)
    loc_start = top_left - np.array([0, size[0]])
    for ind in range(num):
        patches.append(Rectangle(loc_start + ind * loc_diff, size[1], size[0]))
        if ind % 2:
            colors.append(Medium)
        else:
            colors.append(Light)


def add_layer_with_omission(
    patches,
    colors,
    size=(24, 24),
    num=5,
    num_max=8,
    num_dots=4,
    top_left=[0, 0],
    loc_diff=[10, -10],
):
    # add a rectangle
    top_left = np.array(top_left)
    loc_diff = np.array(loc_diff)
    loc_start = top_left - np.array([0, size[0]])
    this_num = min(num, num_max)
    start_omit = (this_num - num_dots) // 2
    end_omit = this_num - start_omit
    start_omit -= 1
    for ind in range(this_num):
        if (num > num_max) and (start_omit < ind < end_omit):
            omit = True
        else:
            omit = False

        if omit:
            patches.append(Circle(loc_start + ind * loc_diff + np.array(size) / 2, 1))
        else:
            patches.append(Rectangle(loc_start + ind * loc_diff, size[1], size[0]))

        if omit:
            colors.append(Black)
        elif ind % 2:
            colors.append(Medium)
        else:
            colors.append(Light)


def add_mapping(
    patches,
    colors,
    start_ratio,
    end_ratio,
    patch_size,
    ind_bgn,
    top_left_list,
    loc_diff_list,
    num_show_list,
    size_list,
):
    start_loc = (
        top_left_list[ind_bgn]
        + (num_show_list[ind_bgn] - 1) * np.array(loc_diff_list[ind_bgn])
        + np.array(
            [
                start_ratio[0] * (size_list[ind_bgn][1] - patch_size[1]),
                -start_ratio[1] * (size_list[ind_bgn][0] - patch_size[0]),
            ]
        )
    )

    end_loc = (
        top_left_list[ind_bgn + 1]
        + (num_show_list[ind_bgn + 1] - 1) * np.array(loc_diff_list[ind_bgn + 1])
        + np.array(
            [
                end_ratio[0] * size_list[ind_bgn + 1][1],
                -end_ratio[1] * size_list[ind_bgn + 1][0],
            ]
        )
    )

    patches.append(Rectangle(start_loc, patch_size[1], -patch_size[0]))
    colors.append(Dark)
    patches.append(Line2D([start_loc[0], end_loc[0]], [start_loc[1], end_loc[1]]))
    colors.append(Darker)
    patches.append(
        Line2D([start_loc[0] + patch_size[1], end_loc[0]], [start_loc[1], end_loc[1]])
    )
    colors.append(Darker)
    patches.append(
        Line2D([start_loc[0], end_loc[0]], [start_loc[1] - patch_size[0], end_loc[1]])
    )
    colors.append(Darker)
    patches.append(
        Line2D(
            [start_loc[0] + patch_size[1], end_loc[0]],
            [start_loc[1] - patch_size[0], end_loc[1]],
        )
    )
    colors.append(Darker)


def label(xy, text, xy_off=[0, 100]):
    plt.text(xy[0] + xy_off[0], xy[1] + xy_off[1], text, family="sans-serif", size=10)


def cria_imagem(model):
    fc_unit_size = 20
    layer_width = 300
    flag_omit = False

    patches = []
    colors = []

    fig, ax = plt.subplots(figsize=(15, 5))

    ############################
    # conv layers
    x_diff_list = [0]
    for l in range(1, len(model.num_list)):
        x_diff_list.append(layer_width)

    text_list = ["Inputs"] + ["Feature\nmaps"] * (len(model.size_list) - 1)
    loc_diff_list = [[5, -5]] * len(model.size_list)

    num_show_list = list(map(min, model.num_list, [NumConvMax] * len(model.num_list)))
    top_left_list = np.c_[np.cumsum(x_diff_list), np.zeros(len(x_diff_list))]

    for ind in range(len(model.size_list) - 1, -1, -1):
        if flag_omit:
            add_layer_with_omission(
                patches,
                colors,
                size=model.size_list[ind],
                num=model.num_list[ind],
                num_max=NumConvMax,
                num_dots=NumDots,
                top_left=top_left_list[ind],
                loc_diff=loc_diff_list[ind],
            )
        else:
            add_layer(
                patches,
                colors,
                size=model.size_list[ind],
                num=num_show_list[ind],
                top_left=top_left_list[ind],
                loc_diff=loc_diff_list[ind],
            )
        label(
            top_left_list[ind],
            text_list[ind]
            + "\n{}@{}x{}".format(
                model.num_list[ind], model.size_list[ind][0], model.size_list[ind][1]
            ),
            xy_off=[0, 30],
        )

    ############################
    # in between layers
    start_ratio_list, end_ratio_list, text_list = [], [], []
    for j in range(1, len(model.num_list)):
        start_ratio_list.append([0.8, 1])
        end_ratio_list.append([0.8, 1])

    for h in range(len(start_ratio_list) // 2):
        text_list.append("Conv")
        text_list.append("Pool")
    ind_bgn_list = range(len(model.patch_size_list))

    for ind in range(len(model.patch_size_list)):
        add_mapping(
            patches,
            colors,
            start_ratio_list[ind],
            end_ratio_list[ind],
            list([0.25 * x for x in model.size_list[ind]]),
            ind,
            top_left_list,
            loc_diff_list,
            num_show_list,
            model.size_list,
        )
        label(
            top_left_list[ind],
            text_list[ind]
            + "\n{}x{}".format(
                model.patch_size_list[ind][0], model.patch_size_list[ind][1]
            ),
            xy_off=[350, -350],
        )

    ############################
    flag_omit = True
    # fully connected layers
    size_list = [(fc_unit_size, fc_unit_size)] * len(model.num_fc_list)
    num_show_list = list(
        map(min, model.num_fc_list, [NumFcMax] * len(model.num_fc_list))
    )
    x_diff_list = [sum(x_diff_list) + layer_width] + [layer_width] * (
        len(model.num_fc_list) - 1
    )
    top_left_list = np.c_[np.cumsum(x_diff_list), np.zeros(len(x_diff_list))]
    loc_diff_list = [[0.8 * fc_unit_size, -fc_unit_size * 0.8]] * len(top_left_list)
    text_list = ["Hidden\nunits"] * (len(size_list) - 1) + ["Outputs"]

    for ind in range(len(size_list)):
        if flag_omit:
            add_layer_with_omission(
                patches,
                colors,
                size=size_list[ind],
                num=model.num_fc_list[ind],
                num_max=NumFcMax,
                num_dots=NumDots,
                top_left=top_left_list[ind],
                loc_diff=loc_diff_list[ind],
            )
        else:
            add_layer(
                patches,
                colors,
                size=size_list[ind],
                num=num_show_list[ind],
                top_left=top_left_list[ind],
                loc_diff=loc_diff_list[ind],
            )
        label(
            top_left_list[ind],
            text_list[ind] + "\n{}".format(model.num_fc_list[ind]),
            xy_off=[-10, 20],
        )

    text_list = ["\n"] + ["Fully\nconnected"] * (len(model.num_fc_list) - 1)

    for ind in range(len(size_list)):
        label(top_left_list[ind], text_list[ind], xy_off=[-250, -350])

    ############################
    for patch, color in zip(patches, colors):
        patch.set_color(color * np.ones(3))
        if isinstance(patch, Line2D):
            ax.add_line(patch)
        else:
            patch.set_edgecolor(Black * np.ones(3))
            ax.add_patch(patch)

    # plt.figure(figsize=(15,8))
    plt.tight_layout()
    plt.axis("equal")
    plt.axis("off")
    plt.show()
    fig.set_size_inches(8, 2.5)
