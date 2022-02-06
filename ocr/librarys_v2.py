from ocr.utill import CTCLabelConverter, AttnLabelConverter
from ocr.ocrmodel import Model
from ocr.dataset import AlignCollate, RawDataset
import torch
import torch.utils.data
import torch.nn.functional as F
import numpy as np
import pandas as pd
import cv2
import pytesseract
import re
from librarys import *


def ocr_predict(opt, images):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    """ model configuration """
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3

    #     print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
    #           opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
    #           opt.SequenceModeling, opt.Prediction)

    model = Model(opt)

    if device == "cuda":
        #         print(device)
        #         model.to(device)
        model = torch.nn.DataParallel(model).to(device)

        model.load_state_dict(torch.load(opt.saved_model, map_location=device))

    else:
        model.load_state_dict(torch.load(opt.saved_model, map_location=device), strict=False)
    # load model
    #     print('loading pretrained model from %s' % opt.saved_model)
    #     model.to(device)

    #     model.load_state_dict(torch.load(opt.saved_model, map_location=device)['model_state_dict'])
    # prepare data. two demo images from https://github.com/bgshih/crnn#run-demo

    AlignCollate_demo = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
    demo_data = RawDataset(root=images, opt=opt)  # use RawDataset

    #     if device=='cuda':
    demo_loader = torch.utils.data.DataLoader(
        demo_data, batch_size=opt.batch_size,
        shuffle=False,
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_demo, pin_memory=True)
    #     else:
    #         demo_loader = torch.utils.data.DataLoader(
    #             demo_data, batch_size=opt.batch_size,
    #             shuffle=False,
    #             num_workers=int(opt.workers),
    #             collate_fn=AlignCollate_demo)
    #     print('데이터 생성완료')

    # predict
    model.eval()
    result_list = []
    with torch.no_grad():
        for image_tensors, image_path_list in demo_loader:
            batch_size = image_tensors.size(0)
            image = image_tensors.to(device)

            # For max length prediction
            length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
            text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)
            #             print('데이터 텐서화')

            if 'CTC' in opt.Prediction:
                preds = model(image, text_for_pred)

                # Select max probabilty (greedy decoding) then decode index to character
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                _, preds_index = preds.max(2)
                preds_index = preds_index.view(-1)
                preds_str = converter.decode(preds_index.data, preds_size.data)

            else:
                preds = model(image, text_for_pred, is_train=False)

                # select max probabilty (greedy decoding) then decode index to character
                _, preds_index = preds.max(2)
                preds_str = converter.decode(preds_index, length_for_pred)

            #             log = open(f'./log_demo_result.txt', 'a')
            dashed_line = '-' * 80
            head = f'{"image_path":25s}\t{"predicted_labels":25s}\tconfidence score'

            #             print(f'{dashed_line}\n{head}\n{dashed_line}')
            #             log.write(f'{dashed_line}\n{head}\n{dashed_line}\n')

            preds_prob = F.softmax(preds, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)
            count = 0

            for img_name, pred, pred_max_prob in zip(image_path_list, preds_str, preds_max_prob):
                count += 1
                if 'Attn' in opt.Prediction:
                    pred_EOS = pred.find('[s]')
                    pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                    pred_max_prob = pred_max_prob[:pred_EOS]

                # calculate confidence score (= multiply of pred_max_prob)
                try:
                    confidence_score = pred_max_prob.cumprod(dim=0)[-1]
                except:
                    confidence_score = 0
                result_list.append([img_name, pred, float(confidence_score)])
    #                 print(f'{img_name}\t{pred:25s}\t{confidence_score:0.4f}')
    #                 log.write(f'{img_name:25s}\t{pred:25s}\t{confidence_score:0.4f}\n')
    #                 img_show(img_name)
    #                 print(pred,confidence_score)

    #             log.close()
    return result_list


def merge_predict(bboxes, result_list):
    # # OCR 결과 Dataframe 생성
    bboxes = np.array(bboxes).astype(int)
    bboxes_df = pd.DataFrame(bboxes.reshape(-1, 8), columns=['x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4'])
    temp = np.array(result_list)
    bboxes_df['ocr'] = temp[:, 1]
    #     bboxes_df['per'] = temp[:, 2]
    merge_df = bboxes_df
    #     merge_df = merging_area(bboxes_df)
    merge_df['height'] = merge_df['y4'] - merge_df['y1']
    merge_df['width'] = merge_df['x2'] - merge_df['x1']
    return merge_df


def merge_predict2(bboxes, result_list, img):
    # # OCR 결과 Dataframe 생성

    bboxes = np.array(bboxes).astype(int)
    bboxes_df = pd.DataFrame(bboxes.reshape(-1, 8), columns=['x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4'])

    temp = np.array(result_list)
    # print(temp)
    bboxes_df['ocr'] = temp
    #     bboxes_df['per'] = temp[:, 2]
    merge_df = bboxes_df.copy()
    default_df = bboxes_df.copy()
    for i in range(5):
        #         print(merge_df)
        merge_df = merging_area(merge_df, img)
        #         if i == 0:
        #             merge_df = merge_df
        bboxes_df = pd.concat([bboxes_df, merge_df])
    #     bboxes_df = pd.concat([bboxes_df,merge_df])

    bboxes_df['height'] = bboxes_df['y4'] - bboxes_df['y1']
    bboxes_df['width'] = bboxes_df['x2'] - bboxes_df['x1']
    default_df['height'] = default_df['y4'] - default_df['y1']
    default_df['width'] = default_df['x2'] - default_df['x1']

    return bboxes_df, default_df


def img_erode(img, kernel_shape, iterations=3):
    kernel = np.ones(kernel_shape, np.uint8)
    img = cv2.erode(img, kernel, iterations=3)
    return img


def merging_area(temp, img):
    temp['left_min_x_p'] = [np.min([x1, x4]) for x1, x4 in temp[['x1', 'x4']].values.tolist()]
    temp['right_max_x_p'] = [np.min([x2, x3]) for x2, x3 in temp[['x2', 'x3']].values.tolist()]
    temp['top_min_y_p'] = [np.min([y1, y2]) for y1, y2 in temp[['y1', 'y2']].values.tolist()]
    temp['bottom_max_y_p'] = [np.max([y4, y3]) for y4, y3 in temp[['y4', 'y3']].values.tolist()]

    ## 오른쪽변 높이 추출
    temp['right_height'] = temp['y3'] - temp['y2']

    temp = temp.sort_values(['y1', 'x1'])
    temp['unique_no'] = temp.index

    for idx, row in temp.iterrows():
        merge_target = temp[(row['right_max_x_p'] <= temp.left_min_x_p)]
        #     print(row)
        merge_target = merge_target[(row['top_min_y_p'] - row['right_height'] / 2) <= merge_target.top_min_y_p]

        merge_target = merge_target[(row['bottom_max_y_p'] + row['right_height'] / 2) >= merge_target.bottom_max_y_p]

        inner_df = merge_target.copy()

        merge_target = merge_target[(row['right_max_x_p'] + row['right_height']) >= merge_target.left_min_x_p]

        #         if len(inner_df.ocr) ==1:
        #             merge_target = pd.concat([merge_target,inner_df])

        if merge_target.shape[0] != 0:

            temp.loc[merge_target.index, 'unique_no'] = row['unique_no']

            temp = temp.sort_index()

        else:

            temp.loc[idx, 'x2'], temp.loc[idx, 'x3'] = row['x2'] + row['right_height'] / 2, row['x3'] + row[
                'right_height'] / 2

    ### 병합된 좌표 추출
    p1 = temp.sort_values(['unique_no', 'left_min_x_p']).groupby('unique_no')[['x1', 'y1', 'x4', 'y4']].head(1).values
    p2 = temp.sort_values(['unique_no', 'left_min_x_p']).groupby('unique_no')[['x2', 'y2', 'x3', 'y3']].tail(1).values
    ocr_df = pd.DataFrame(np.concatenate((p1, p2), axis=1), columns=['x1', 'y1', 'x4', 'y4', 'x2', 'y2', 'x3', 'y3'])

    ### ocr 병합
    ocr_df['ocr'] = temp.sort_values(['unique_no', 'left_min_x_p']).groupby('unique_no').ocr.apply(
        lambda x: ' '.join(x)).values
    ocr_df = ocr_df[['x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4', 'ocr']]
    ### 정렬

    return ocr_df


def set_imgs_margin(imgs, fill_cnt=40, is_color=False):
    h_fill_cnt = int(fill_cnt / 2)
    new_imgs = []
    len(imgs)
    for i in imgs:
        if is_color:
            new_bg_img = np.zeros([i.shape[0] + fill_cnt, i.shape[1] + fill_cnt, 3], dtype='uint8')
            new_bg_img.fill(255)
            new_bg_img[h_fill_cnt:(h_fill_cnt * -1), h_fill_cnt:(h_fill_cnt * -1)] = i
            new_imgs.append(new_bg_img)
        else:
            new_bg_img = np.zeros([i.shape[0] + fill_cnt, i.shape[1] + fill_cnt], dtype='uint8')
            new_bg_img.fill(255)
            new_bg_img[h_fill_cnt:(h_fill_cnt * -1), h_fill_cnt:(h_fill_cnt * -1)] = i
            new_imgs.append(new_bg_img)
    return new_imgs


# def get_ocr_data(img, output_type='string', is_gray_scaler=True, debug=True, is_color=False, processing_type=1):
#     # tessdata_dir = '--tessdata-dir ' + '/usr/share/tesseract-ocr/4.00/tessdata --psm 6'
#     #     tessdata_dir = '--tessdata-dir /Users/soncheoljun/ocr_api/ocsapi/tessdata/40 --psm 6'
#     tessdata_dir = '--tessdata-dir ' + '/workspace/DBP/서류ocr인식/tessdata --psm 1'
#     ori_img = img.copy()
#
#     if is_gray_scaler:
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     #         img = gray_scale(img)
#
#     if processing_type == 1:
#         size = 100
#         r = size / img.shape[0]
#         dim = (int(img.shape[1] * r), size)
#         if dim[0] != 0 and dim[1] != 0:
#             img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
#         img = img_erode(img, 2)
#     elif processing_type == 2:
#         kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
#         border = cv2.copyMakeBorder(img, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=[255, 255])
#         resizing = cv2.resize(border, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
#         dilation = cv2.dilate(resizing, kernel, iterations=1)
#         erosion = cv2.erode(dilation, kernel, iterations=1)
#
#     if is_color == False:
#         img = set_imgs_margin([img], is_color=False)[0]
#
#     if output_type == 'dataframe':
#         d = pytesseract.image_to_data(img, lang='kor', output_type=Output.DICT, config=tessdata_dir)
#         return_data = pd.DataFrame.from_dict(d)
#         string = return_data['text'][0]
#     elif output_type == 'string':
#         string = pytesseract.image_to_string(img, lang='kor', config=tessdata_dir)
#         return_data = string
#     elif output_type == 'list':
#         d = pytesseract.image_to_data(img, lang='kor', output_type='data.frame', config=tessdata_dir)
#         d = d[d.conf != -1]
#
#         lines = d.groupby('block_num')['text'].apply(lambda x: ''.join(x)).tolist()
#         conf = d.groupby(['block_num'])['conf'].mean().tolist()
#         return_data = [lines, conf]
#
#     if debug:
#         img_show(img)
#         # print(string)
#
#     return return_data


def convert(test_keyword):
    split_keyword_list = list(test_keyword)
    # 유니코드 한글 시작 : 44032, 끝 : 55199
    BASE_CODE, CHOSUNG, JUNGSUNG = 44032, 588, 28
    # 초성 리스트. 00 ~ 18
    CHOSUNG_LIST = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
    # 중성 리스트. 00 ~ 20
    JUNGSUNG_LIST = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ',
                     'ㅣ']
    # 종성 리스트. 00 ~ 27 + 1(1개 없음)
    JONGSUNG_LIST = [' ', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ',
                     'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

    result = list()
    for keyword in split_keyword_list:
        # 한글 여부 check 후 분리
        if re.match('.*[ㄱ-ㅎㅏ-ㅣ가-힣]+.*', keyword) is not None:

            char_code = ord(keyword) - BASE_CODE

            if char_code >= 0:
                char1 = int(char_code / CHOSUNG)

                result.append(CHOSUNG_LIST[char1])

                #             print('result1',result)
                char2 = int((char_code - (CHOSUNG * char1)) / JUNGSUNG)
                result.append(JUNGSUNG_LIST[char2])
                #             print('result2',result)
                char3 = int((char_code - (CHOSUNG * char1) - (JUNGSUNG * char2)))
                if char3 == 0:
                    continue
                #                 result.append('#')
                else:
                    result.append(JONGSUNG_LIST[char3])
        #             print('result3',result)

        else:
            result.append(keyword)
    return result


def char_error_cnt(target_text, text, return_type='cnt'):
    #     text = text.replace(' ', '')

    docompose_target_text = convert(target_text)
    s_decompose_text = convert(text)

    t_size = len(docompose_target_text)
    if t_size == 0:
        return 0
    len_diff = np.abs(t_size - len(s_decompose_text))
    cer_error_cnt_list = []
    if (t_size - len(s_decompose_text)) > 0:

        for i in range(len_diff + 1):

            sample_text = s_decompose_text
            sample_text += [''] * (len(docompose_target_text) - len(
                s_decompose_text) - i)  # for i in range(len(docompose_target_text) - len(s_decompose_text)-c)]
            #             print(sample_text)
            if i > 0:
                sample_text.insert(0, '')
                sample_text.pop(-1)
            #             print(sample_text,docompose_target_text)
            cer_error_cnt = (np.array(sample_text) == np.array(docompose_target_text)).sum()
            cer_error_cnt_list.append(cer_error_cnt)



    elif (t_size - len(s_decompose_text)) == 0:

        sample_text = s_decompose_text

        cer_error_cnt = (np.array(sample_text) == np.array(docompose_target_text)).sum()
        cer_error_cnt_list.append(cer_error_cnt)

    else:

        for i in range(len_diff + 1):
            sample_text = docompose_target_text
            sample_text += [''] * (len(s_decompose_text) - len(
                docompose_target_text) - i)  # for i in range(len(docompose_target_text) - len(s_decompose_text)-c)]

            if i > 0:
                sample_text.insert(0, '')
                sample_text.pop(-1)

            cer_error_cnt = (np.array(sample_text) == np.array(s_decompose_text)).sum()
            cer_error_cnt_list.append(cer_error_cnt)

    if return_type == 'cnt':
        return np.max(cer_error_cnt_list)

    if return_type == 'per':
        if (t_size - len(s_decompose_text)) < 0:
            return np.max(cer_error_cnt_list) / len(s_decompose_text)
        else:

            return np.max(cer_error_cnt_list) / t_size


def char_error_cnt2(target_text, text, return_type='cnt'):
    #     text = text.replace(' ', '')

    docompose_target_text = convert(target_text)
    s_decompose_text = convert(text)

    t_size = len(docompose_target_text)
    if t_size == 0:
        return 0

    if len(s_decompose_text) < t_size:
        return 0

    len_diff = np.abs(t_size - len(s_decompose_text))
    cer_error_cnt_list = []

    for i in range(len_diff + 1):

        sample_text = docompose_target_text
        sample_text += [''] * (len(s_decompose_text) - len(
            docompose_target_text) - i)  # for i in range(len(docompose_target_text) - len(s_decompose_text)-c)]

        if i > 0:
            sample_text.insert(0, '')
            sample_text.pop(-1)

        cer_error_cnt = (np.array(sample_text) == np.array(s_decompose_text)).sum()

        cer_error_cnt_list.append(cer_error_cnt)

    if return_type == 'cnt':
        return np.max(cer_error_cnt_list)

    if return_type == 'per':
        return np.max(cer_error_cnt_list) / t_size
