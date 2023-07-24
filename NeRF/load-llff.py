import genericpath

import numpy as np
import os, imageio

def _minify(basedir, factors=[], resolution=[]):
    # 判断是否需要读取
    needtoload = False
    for r in factors:
        imgdir = os.path.join(basedir, 'image_{}'.format(r))
        if not os.path.exists(imgdir):
            needtoload = True
    for r in resolution:
        imgdir = os.path.join(basedir, 'image_{}x{}'.format(r[1], r[0]))
        if not genericpath.exists(imgdir):
            needtoload = True
    if not needtoload:
        return

    # 文件操作模块
    from shutil import copy
    # 管理子进程
    from subprocess import  check_output

    imgdir = os.path.join(basedir, 'images')
    imgs = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))]  # 按字母序列排序
    imgs = [f for f in imgs if any([f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])]  # 后缀
    imgdir_orig = imgdir

    cwd = os.getcwd()

    for r in factors + resolution:
        # 判断是否为整数
        if isinstance(r, int):
            name = 'image_{}'.format(r)
            resizearg = '{}%'.format(100./r)
        else:
            name = 'images_{}x{}'.format(r[1], r[0])
            resizearg = '{}x{}'.format(r[1], r[0])
        # 判断是否存在imgdir
        imgdir = os.path.join(basedir, name)
        if os.path.exists(imgdir):
            continue

        print('Minifying', r, basedir)

        os.makedirs(imgdir)
        # 将文件从原始目录复制到新目录
        check_output('cp {}/* {}'.format(imgdir_orig, imgdir), shell=True)

        # 获取扩展名
        ext = imgs[0].split('.')[-1]
        args = ' '.join(['mogrify', '-resize', resizearg, '-format', 'png', '*.{}'.format(ext)])
        print(args)
        os.chdir(imgdir)
        check_output(args, shell=True)  # 执行args
        os.chdir(cwd)

        # 如果不是则删除
        if ext != 'png':
            check_output('rm {}/*.{}'.format(imgdir, ext), shell=True)
            print('Removed duplicates')
        print('Done')
































