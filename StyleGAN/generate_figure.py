
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
 
"""Minimal script for reproducing the figures of the StyleGAN paper using pre-trained generators."""

import os
import pickle
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import config
import glob


#----------------------------------------------------------------------------
# Helpers for loading and using pre-trained generators.
 
# pre-trained network.
Model = './cache/2019-03-08-stylegan-animefaces-network-02051-021980.pkl'

 
synthesis_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True), minibatch_size=8)
_Gs_cache = dict()

# 加载StyleGAN已训练好的网络模型
def load_Gs(model):
    if model not in _Gs_cache:
        model_file = glob.glob(Model)
        if len(model_file) == 1:
            model_file = open(model_file[0], "rb")
        else:
            raise Exception('Failed to find the model')
 
        _G, _D, Gs = pickle.load(model_file)
        # _G = Instantaneous snapshot of the generator. Mainly useful for resuming a previous training run.
        # _D = Instantaneous snapshot of the discriminator. Mainly useful for resuming a previous training run.
        # Gs = Long-term average of the generator. Yields higher-quality results than the instantaneous snapshot.
 
        # Print network details.
        Gs.print_layers()
 
        _Gs_cache[model] = Gs
    return _Gs_cache[model]


#----------------------------------------------------------------------------
# Figures 2, 3, 10, 11, 12: Multi-resolution grid of uncurated result images.
# lod, level of detail
# 多层次细节处理，在相机距离不同的情况下，使得物体显示不同的模型，从而节省性能的开销。
# 这里的实际处理是把图片按比例缩小，按照不同大小的网格显示随机生成的图片，使得效果更炫一些。
 
def draw_uncurated_result_figure(png, Gs, cx, cy, cw, ch, rows, lods, seed):
    print(png)
    # 规划在网格中显示的图片的数量，按此数量定义latents的数量
    latents = np.random.RandomState(seed).randn(sum(rows * 2**lod for lod in lods), Gs.input_shape[1])
    # 使用Gs.run()直接生成输出为numpy数组的图像
    images = Gs.run(latents, None, **synthesis_kwargs) # [seed, y, x, rgb]
 
    # 绘制空白画布
    canvas = PIL.Image.new('RGB', (sum(cw // 2**lod for lod in lods), ch * rows), 'white')
    # iteration（）产生一个迭代器，使用next()方法获取下一个项
    image_iter = iter(list(images))
    # 在画布的网格中逐一绘制生成的图像
    for col, lod in enumerate(lods):
        for row in range(rows * 2**lod):
            image = PIL.Image.fromarray(next(image_iter), 'RGB')
            image = image.crop((cx, cy, cx + cw, cy + ch))
            image = image.resize((cw // 2**lod, ch // 2**lod), PIL.Image.ANTIALIAS)
            canvas.paste(image, (sum(cw // 2**lod for lod in lods[:col]), row * ch // 2**lod))
    canvas.save(png)
#----------------------------------------------------------------------------
# Figure 3: Style mixing.
# 分别用不同的种子生成源图像和目标图像，然后用源图像的src_dlatents的一部分替换目标图像的dst_dlatents的对应部分，
# 然后用Gs.components.synthesis.run（）函数生成风格混合后的图像
 
def draw_style_mixing_figure(png, Gs, w, h, src_seeds, dst_seeds, style_ranges):
    print(png)
 
    # 生成随机的latents，包括：src_latents和dst_latents，都是Gs.input_shape[1]大小的张量，对512x512的图片来说，就是512
    # 重复生成，src_latents共有len(src_seeds)个张量（主程序中设定为5），dst_latents共有len(dst_seeds)个张量（主程序中设定为6）
    src_latents = np.stack(np.random.RandomState(seed).randn(Gs.input_shape[1]) for seed in src_seeds)
    dst_latents = np.stack(np.random.RandomState(seed).randn(Gs.input_shape[1]) for seed in dst_seeds)
 
    # 按照StyleGAN的网络架构，从z变换到w，对于512x512的图片来说，src_dlatents的shape是5x16x512，dst_dlatents的shape是6x16x512
    src_dlatents = Gs.components.mapping.run(src_latents, None) # [seed, layer, component]
    dst_dlatents = Gs.components.mapping.run(dst_latents, None) # [seed, layer, component]
 
    # 从w生成图像
    src_images = Gs.components.synthesis.run(src_dlatents, randomize_noise=False, **synthesis_kwargs)
    dst_images = Gs.components.synthesis.run(dst_dlatents, randomize_noise=False, **synthesis_kwargs)
 
    # 画空白画布
    canvas = PIL.Image.new('RGB', (w * (len(src_seeds) + 1), h * (len(dst_seeds) + 1)), 'white')
 
    # 在画布的第一行画源图像，第一格空白
    for col, src_image in enumerate(list(src_images)):
        canvas.paste(PIL.Image.fromarray(src_image, 'RGB'), ((col + 1) * w, 0))
 
    # 在画布逐行绘制图像
    for row, dst_image in enumerate(list(dst_images)):
 
        # 首列绘制目标图像
        canvas.paste(PIL.Image.fromarray(dst_image, 'RGB'), (0, (row + 1) * h))
 
        # 将目标图像复制src_seeds份（主程序中设定为5），构成新数组
        row_dlatents = np.stack([dst_dlatents[row]] * len(src_seeds))
 
        # 用src_dlatents的指定列替换row_dlatents的指定列，数据混合
        row_dlatents[:, style_ranges[row]] = src_dlatents[:, style_ranges[row]]
 
        # 调用用Gs.components.synthesis.run（）函数生成风格混合后的图像
        row_images = Gs.components.synthesis.run(row_dlatents, randomize_noise=False, **synthesis_kwargs)
 
        # 在画布上逐列绘制风格混合后的图像
        for col, image in enumerate(list(row_images)):
            canvas.paste(PIL.Image.fromarray(image, 'RGB'), ((col + 1) * w, (row + 1) * h))
 
    canvas.save(png)
 
#----------------------------------------------------------------------------
# Figure 4: Noise detail.
# 以计算统计均方差的方式展现图片生成过程中的噪声的影响
 
def draw_noise_detail_figure(png, Gs, w, h, num_samples, seeds):
    print(png)
 
    # 画布为3列，一共len(seeds)行
    canvas = PIL.Image.new('RGB', (w * 3, h * len(seeds)), 'white')
 
    # 逐行画图
    for row, seed in enumerate(seeds):
        # latents是大小为len(Gs.input_shape[1]的张量，默认为512，相同的种子，一次生成num_samples个张量（主程序设定为100）
        latents = np.stack([np.random.RandomState(seed).randn(Gs.input_shape[1])] * num_samples)
        # 允许生成器使用“截断技巧”（truncation trick），通过通过设置阈值的方式来截断 z 的采样，提高图片的生成质量（但同时可能会降低生成图片的差异性）
        # 使用Gs.run()直接生成输出为numpy数组的图像
        images = Gs.run(latents, None, truncation_psi=1, **synthesis_kwargs)
 
        # 画图，第一列
        canvas.paste(PIL.Image.fromarray(images[0], 'RGB'), (0, row * h))
 
        # 剪裁，放大image1、2、3、4，并画图，让你看看图片细节上的某些差异（即：噪声）
        for i in range(4):
            crop = PIL.Image.fromarray(images[i + 1], 'RGB')
            # 我们的图是512x512，所以截取的区域都除以2
            crop = crop.crop((650/2, 180/2, 906/2, 436/2))
            crop = crop.resize((w//2, h//2), PIL.Image.NEAREST)
            canvas.paste(crop, (w + (i%2) * w//2, row * h + (i//2) * h//2))
 
        # 对所有图像的同一个像素（x，y)的数值先计算均值，然后在每一行上计算标准差，最后乘以4
        diff = np.std(np.mean(images, axis=3), axis=0) * 4
        # 0-255内取值，超出取值范围的按边界值取值，四舍五入
        diff = np.clip(diff + 0.5, 0, 255).astype(np.uint8)
 
        # 在第三列画图，即多图计算方差后得出的“噪声”
        canvas.paste(PIL.Image.fromarray(diff, 'L'), (w * 2, row * h))
    canvas.save(png)
 
#----------------------------------------------------------------------------
# Figure 5: Noise components.
# 显示不同的噪声区间对生成图片的影响
 
def draw_noise_components_figure(png, Gs, w, h, seeds, noise_ranges, flips):
    print(png)
 
    # 创建Gs网络的一个克隆体，包括所有的变量
    Gsc = Gs.clone()
 
    # vars() 函数返回对象的属性和属性值的字典对象，这里返回的是随机生成的noise_input[]。
    noise_vars = [var for name, var in Gsc.components.synthesis.vars.items() if name.startswith('noise')]
    # tflib.run()运行网络，为noise_vars赋值
    noise_pairs = list(zip(noise_vars, tflib.run(noise_vars))) # [(var, val), ...]
    # 为Gsc创建初始张量latents，主程序给定2个种子
    latents = np.stack(np.random.RandomState(seed).randn(Gs.input_shape[1]) for seed in seeds)
 
    all_images = []
    # 添加不同噪声后生成图片
    for noise_range in noise_ranges:
        # 赋值，在噪声区间内的值保留，否则置0
        tflib.set_vars({var: val * (1 if i in noise_range else 0) for i, (var, val) in enumerate(noise_pairs)})
        # 给定噪声区间，用Gsc生成不同的图片
        range_images = Gsc.run(latents, None, truncation_psi=1, randomize_noise=False, **synthesis_kwargs)
        range_images[flips, :, :] = range_images[flips, :, ::-1]
        all_images.append(list(range_images))
 
    # 绘制空白画布
    canvas = PIL.Image.new('RGB', (w * 2, h * 2), 'white')
    for col, col_images in enumerate(zip(*all_images)): # col = 2，两个种子，生成两组图片
        # 画图，第一组图片放在第一列，第二组图片放在第二列
        # image[0]左半边和image[1]右半边画在同一列的第一行，image[2]的左半边和image[3]的右半边画在同一列的第二行
        canvas.paste(PIL.Image.fromarray(col_images[0], 'RGB').crop((0, 0, w//2, h)), (col * w, 0))
        canvas.paste(PIL.Image.fromarray(col_images[1], 'RGB').crop((w//2, 0, w, h)), (col * w + w//2, 0))
        canvas.paste(PIL.Image.fromarray(col_images[2], 'RGB').crop((0, 0, w//2, h)), (col * w, h))
        canvas.paste(PIL.Image.fromarray(col_images[3], 'RGB').crop((w//2, 0, w, h)), (col * w + w//2, h))
    canvas.save(png)
 
#----------------------------------------------------------------------------
# Figure 8: Truncation trick.
# 展现不同截断阈值对生成图片的影响
 
def draw_truncation_trick_figure(png, Gs, w, h, seeds, psis):
    print(png)
 
    # 随机生成初始张量latents
    latents = np.stack(np.random.RandomState(seed).randn(Gs.input_shape[1]) for seed in seeds)
    # z映射到w
    dlatents = Gs.components.mapping.run(latents, None) # [seed, layer, component]
 
    # 获取w的均值dlatent_avg
    dlatent_avg = Gs.get_var('dlatent_avg') # [component]
 
    # 绘制空白画布，共2行，每行展示同一种子不同截断阈值下生成的图片
    canvas = PIL.Image.new('RGB', (w * len(psis), h * len(seeds)), 'white')
    for row, dlatent in enumerate(list(dlatents)):
        # 将dlatent张量增加维度，依照主程序设定的截断阈值的个数复制了N份，用截断阈值的比例大小控制row_dlatents
        # row_dlatents满足下面synthesis.run()的输入数组的维度要求
        row_dlatents = (dlatent[np.newaxis] - dlatent_avg) * np.reshape(psis, [-1, 1, 1]) + dlatent_avg
        # 将被截断阈值调控后的row_dlatents用于StyleGAN网络模型
        row_images = Gs.components.synthesis.run(row_dlatents, randomize_noise=False, **synthesis_kwargs)
 
        # 在画布上画图
        for col, image in enumerate(list(row_images)):
            canvas.paste(PIL.Image.fromarray(image, 'RGB'), (col * w, row * h))
    canvas.save(png)

 
#----------------------------------------------------------------------------
# Main program.
 
def main():
    tflib.init_tf()
    os.makedirs(config.result_dir, exist_ok=True)
 
    # 将许多未经筛选的图片集中绘制在一起，按不同的缩小比例布置在画布上
    #draw_uncurated_result_figure(os.path.join(config.result_dir, 'figure02-uncurated-animation.png'), load_Gs(Model), cx=0, cy=0, cw=512, ch=512, rows=3, lods=[0, 1, 2, 2, 3, 3], seed=5)
 
    # # 这里除了要把w、h从1024修改为512以外，还得把range(8, 18)修改为range（8，16），因为StyleGAN在生成1024x1024的图片时的合成网络g是18层的，生成512x512图片时的合成网络g是16层的
    # # 精心选择的种子(src_seeds, dst_seeds)，应该与生成图片的效果有关
    draw_style_mixing_figure(os.path.join(config.result_dir, 'figure03-style-mixing.png'), load_Gs(Model), w=512, h=512, src_seeds=[639, 701, 687, 615, 2268], dst_seeds=[888, 829, 1898, 1733, 1614, 845], style_ranges=[range(0, 4)] * 3 + [range(4, 8)] * 2 + [range(8, 16)])
 
    # # 自动生成的图片，展现在种子相同时不同图片之间的噪声
    # draw_noise_detail_figure(os.path.join(config.result_dir, 'figure04-noise-detail.png'), load_Gs(Model), w=512, h=512, num_samples=100, seeds=[1157, 1012])
 
    # # 给定两个种子，设置4个噪声区间，显示不同噪声水平下的图片内容
    # draw_noise_components_figure(os.path.join(config.result_dir, 'figure05-noise-components.png'), load_Gs(Model), w=512, h=512, seeds=[1967, 1555], noise_ranges=[range(0, 16), range(0, 0), range(8, 16), range(0, 8)], flips=[1])
 
    # # 给定两个种子，6个truncation_psi，用于比较不同“截断技巧”（truncation trick）水平下的图片生成质量和差异度
    # draw_truncation_trick_figure(os.path.join(config.result_dir, 'figure08-truncation-trick.png'), load_Gs(Model), w=512, h=512, seeds=[91, 388], psis=[1, 0.7, 0.5, 0, -0.5, -1])
 
#----------------------------------------------------------------------------
 
if __name__ == "__main__":
    main()
 


