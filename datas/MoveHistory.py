import os
import copy
import json
import glob

import numpy as np

class MoveHistory:

    def __init__(self):

        self.state = []         # 現在の状態
        self.targets = []       # 選択可能な行動
        self.action = -1        # 選択した行動
        self.reward = 0         # 報酬 
        self.state_1 = []       # 行動した後の状態
        self.targets_1 = []     # 行動した後の選択可能な行動
        self.terminal = False   # 終わりかどうか

    def remove_play_data(file_name):

        # ファイル名
        play_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "self_play_datas")
        path = os.path.join(play_data_dir, '{}-*.json'.format(file_name))

        list_json = glob.glob(path)  
        
        for path in list_json:
            os.remove(path)


    def load_play_data(file_name):
        move_list = []

        # ファイル名
        play_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "self_play_datas")
        path = os.path.join(play_data_dir, '{}-*.json'.format(file_name))

        list_json = glob.glob(path)

        for path in list_json:

            with open(path, "r") as f:
                fstr = f.read()  # ファイル終端まで全て読んだデータを返す

            js = json.loads(fstr) 

            for j in js:
                buffer = MoveHistory()

                tmp = j['state']
                buffer.state = np.array(tmp)
                buffer.targets = j['targets']
                buffer.action = j['action']
                buffer.reward = j['reward']
                buffer.terminal =  j['terminal']
                if buffer.terminal == False:
                    tmp = j['state_1']
                    buffer.state_1 = np.array(tmp)
                    buffer.targets_1 = j['targets_1']

                move_list.append(buffer)

        return move_list


    def save_play_data(file_name, move_list: list):

        # ファイル名
        play_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "self_play_datas")

        # 保存用のデータに整形する
        list_json = []
        for tmp in move_list:
            dict_json = dict()
            dict_json['state'] = tmp.state.tolist()                     # 現在の状態
            dict_json['targets'] = list(map(int, tmp.targets))          # 選択可能な行動
            dict_json['action'] = int(tmp.action)                       # 選択した行動
            dict_json['reward'] = float(tmp.reward)                     # 報酬 
            dict_json['terminal'] = tmp.terminal                        # 終わりかどうか
            # 終わりじゃなかったら次の状態を保存する
            if tmp.terminal == False:
                dict_json['state_1'] = tmp.state_1.tolist()             # 行動した後の状態
                dict_json['targets_1'] = list(map(int, tmp.targets_1))  # 行動した後の選択可能な行動
            else:
                dict_json['state_1'] = []
                dict_json['targets_1'] = []

            list_json.append(dict_json)

        # 大きすぎるJSON ファイルは NG なので データ10 個毎に保存する
        i = 0
        idx = 0
        sub_list = []
        for tmp in list_json:
            sub_list.append(tmp)
            i += 1
            if i % 10 == 0 or i == len(list_json):
                idx += 1
                path = os.path.join(play_data_dir, '{}-{}.json'.format(file_name, idx))
                # Json 形式で、保存
                with open(path, "a") as f:
                    json.dump(sub_list, f)
                sub_list = []



