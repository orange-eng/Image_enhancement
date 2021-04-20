
import os
import pickle
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import config
import glob
import cv2 as cv
import os
import sys
path = os.path.abspath(os.path.dirname(sys.argv[0]))

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

def draw_style_mixing_figure(png, Gs, w, h, src_seeds, dst_seeds, style_ranges):
    # 生成随机的latents，包括：src_latents和dst_latents，都是Gs.input_shape[1]大小的张量，对512x512的图片来说，就是512
    # 重复生成，src_latents共有len(src_seeds)个张量（主程序中设定为5），dst_latents共有len(dst_seeds)个张量（主程序中设定为6）
    src_latents = np.stack(np.random.RandomState(seed).randn(Gs.input_shape[1]) for seed in src_seeds)      #(5,512)
    dst_latents = np.stack(np.random.RandomState(seed).randn(Gs.input_shape[1]) for seed in dst_seeds)      #(6,512)
    # 按照StyleGAN的网络架构，从z变换到w，对于512x512的图片来说，src_dlatents的shape是5x16x512，dst_dlatents的shape是6x16x512
    src_dlatents = Gs.components.mapping.run(src_latents, None) # [seed, layer, component]
    dst_dlatents = Gs.components.mapping.run(dst_latents, None) # [seed, layer, component]

    # 从w生成图像
    src_images = Gs.components.synthesis.run(src_dlatents, randomize_noise=False, **synthesis_kwargs)
    dst_images = Gs.components.synthesis.run(dst_dlatents, randomize_noise=False, **synthesis_kwargs)

    orange_img = cv.imread(path + "/img/orange.jpg")
    catong_img = cv.imread(path + "/img/catong.jpg")

    orange_img =  cv.resize(orange_img,(512,512))
    catong_img =  cv.resize(catong_img,(512,512))

    orange_img = orange_img.reshape((1,512,512,3))
    catong_img = catong_img.reshape((1,512,512,3))
    print(orange_img.shape)
    print(type(orange_img))
    print(src_images.shape)
    print(type(src_images))
    # # 画空白画布
    # canvas = PIL.Image.new('RGB', (w * (len(src_seeds) + 1), h * (len(dst_seeds) + 1)), 'white')
    # # 在画布的第一行画源图像，第一格空白
    # for col, src_image in enumerate(list(src_images)):  # 5次循环
    #     canvas.paste(PIL.Image.fromarray(src_image, 'RGB'), ((col + 1) * w, 0))
    # # 在画布逐行绘制图像
    # for row,dst_image in enumerate(list(dst_images)):   #6次循环
    #     canvas.paste(PIL.Image.fromarray(dst_image,'RGB'),(0,(row+1)*h))
    #     # 及那个目标图像复制scr_Seeds份，构建新的数组
    #     row_dlatents = np.stack([dst_dlatents[row]]*len(src_seeds))
    #     # 用src_dlatents的制定列替换row_dlatents的指定列，数据融合
    #     row_dlatents[:,style_ranges[row]] = src_dlatents[:,style_ranges[row]]
    #     # 调用Gs.components.synthesis.run()函数生成风格混合后的图像
    #     row_images = Gs.component.synthesis.run(row_dlatents,randomize_noise=False,**synthesis_kwargs)
    #     #在画布上绘制混合之后的图像
    #     for col,image in enumerate(list(row_images)):   #6次循环
    #         canvas.paste(PIL.Image.fromarray(image,'RGB'),((col+1)*w,(row+1)*h))
    # canvas.save(png)

if __name__ == "__main__":
    tflib.init_tf()
    os.makedirs(config.result_dir, exist_ok=True)
    # # 这里除了要把w、h从1024修改为512以外，还得把range(8, 18)修改为range（8，16），因为StyleGAN在生成1024x1024的图片时的合成网络g是18层的，生成512x512图片时的合成网络g是16层的
    # # 精心选择的种子(src_seeds, dst_seeds)，应该与生成图片的效果有关
    draw_style_mixing_figure(os.path.join(config.result_dir, 'figure03-style-mixing.png'), load_Gs(Model), w=512, h=512, src_seeds=[639], dst_seeds=[888, 829, 1898, 1733, 1614, 845], style_ranges=[range(0, 4)] * 3 + [range(4, 8)] * 2 + [range(8, 16)])
