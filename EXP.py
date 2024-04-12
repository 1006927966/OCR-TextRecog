import os
import shutil
import re
# from models.BuildModel import crnnModel, SVTRModel
# import yaml
#
# config = yaml.load(open("/code/wujilong/code/OCR-TextRecog/config/SVTR-T-Ch-stn.yaml", 'r', encoding='utf-8'), Loader=yaml.FullLoader)
#
# model = SVTRModel(config).cuda()
#
# print(model)


def readlines(savedir):
    os.makedirs(savedir, exist_ok=True)
    srcdir = "/code/wangshiyuan02/data/ocr/commocr/textline/generateData/companyname700w40032"
    txtpath = "/code/wangshiyuan02/data/ocr/commocr/textline/generateData/tmp_labels700w.txt"

    picpaths = []
    labels = []
    count = 0
    with open(txtpath, "r", encoding="utf-8") as f:
        line = f.readline()
        while line:
            if "_" not in line:
                line = f.readline()
                continue
            if count > 105:
                break
            count += 1
            factors = line.strip().split()
            picname = factors[0] + ".jpg"
            label = factors[-1]
            picpath = os.path.join(srcdir, picname)
            if os.path.exists(picpath):
                dstpath = os.path.join(savedir, label+".jpg")
                shutil.copy(picpath, dstpath)
                picpaths.append(picpath)
                labels.append(label)
            line = f.readline()
    return picpaths, labels


def calculate(txtpath):
    with open(txtpath, "r") as f:
        lines = f.readlines()
    allnum = len(lines)
    count = 0
    for line in lines:
        factors = line.strip().split("||")
        keyname = os.path.split(factors[0])[-1][:-4]
        # print(keyname)
        label = factors[1]
        label = re.sub(r"(限)+", "限", label)
        label = re.sub(r"(责)+", "责", label)
        label = re.sub(r"(任)+", "任", label)
        label = label.replace("公公司", "公司")
        if keyname == label:
            count += 1
    return count / allnum
crnntxt = "/code/wujilong/code/OCR-TextRecog/crnn.txt"
svtrtxt = "/code/wujilong/code/OCR-TextRecog/svtr.txt"

crnnrecall = calculate(crnntxt)
svtrrecall = calculate(svtrtxt)
print("crnn {}".format(crnnrecall))
print("svtr {}".format(svtrrecall))