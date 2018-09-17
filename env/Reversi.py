import os
import numpy as np

class Reversi:

    def __init__(self):
        # parameters
        self.name = os.path.splitext(os.path.basename(__file__))[0]
        self.Blank = 0
        self.Black = 1
        self.White = 2
        self.next_player = self.Black
        self.screen_n_rows = 8
        self.screen_n_cols = 8
        self.enable_actions = np.arange(self.screen_n_rows*self.screen_n_cols)
        # variables
        self.reset()
        

    def reset(self):
        """ 盤面の初期化 """
        # reset ball position
        self.screen = np.zeros((self.screen_n_rows, self.screen_n_cols))
        self.set_cells(27, self.White)
        self.set_cells(28, self.Black)
        self.set_cells(35, self.Black)
        self.set_cells(36, self.White)

        self.next_player = self.Black


    def get_cells(self, i):
        r = int(i / self.screen_n_cols)
        c = int(i - ( r * self.screen_n_cols))
        return self.screen[r][c]  
        
       
    def set_cells(self, i, value):
        r = int(i / self.screen_n_cols)
        c = int(i - ( r * self.screen_n_cols))
        self.screen[r][c] = value 
      
      
    def print_screen(self):
        """ 盤面の出力 """
        i = 0
        for r in range(self.screen_n_rows):
            s1 = ''
            for c in range(self.screen_n_cols):
                s2 = ''
                if self.screen[r][c] == self.Blank:
                    s2 = '{0:2d}'.format(self.enable_actions[i])
                elif self.screen[r][c] == self.Black:
                    s2 = '●'
                elif self.screen[r][c] == self.White:
                    s2 = '○'
                s1 = s1 + ' ' + s2
                i += 1
            print(s1)


    def put_piece(self, action, color, puton=True):
        """自駒color(1 or 2)を位置action(0～63)に置く関数 """
         
        if self.get_cells(action) != self.Blank:
            return -1

        """ ---------------------------------------------------------
           縦横斜めの8通りは、1次元データなので、
           現在位置から[-9, -8, -7, -1, 1, 7, 8, 9] 
           ずれた方向を見ます。
           これは、[-1, 0, 1]と[-8, 0, 8]の組合せで調べます
           (0と0のペアは除く)。
        """
        t, x, y, l = 0, action%8, action//8, []
        for di, fi in zip([-1, 0, 1], [x, 7, 7-x]):
            for dj, fj in zip([-8, 0, 8], [y, 7, 7-y]):
                
                if not di == dj == 0:
                    b, j, k, m, n =[], 0, 0, [], 0                    
                    """a:対象位置のid リスト"""
                    a = self.enable_actions[action+di+dj::di+dj][:min(fi, fj)]
                    """b:対象位置の駒id リスト"""
                    for i in a: 
                        b.append(self.get_cells(i))
                    
                    for i in b:
                        if i == 0: #空白
                            break  
                        elif i == color: #自駒があればその間の相手の駒を取れる
                            """ 取れる数を確定する """ 
                            n = k
                            """ ひっくり返す駒を確定する """ 
                            l += m
                            """ その方向の探査終了 """
                            break
                        else: #相手の駒
                            k += 1
                            """ ひっくり返す位置をストックする """ 
                            m.insert(0, a[j]) 
                        j += 1
                    t += n 
            
        if puton == True: 
            #実際にひっくり返してゲームを進行させる

            """ ひっくり返す石を登録する """
            for i in l:
                self.set_cells(i, color)

            """ 今置いた石を追加する """ 
            self.set_cells(action, color)

        return t


        
    def winner(self):
        """ 勝ったほうを返す """
        Black_score = self.get_score(self.Black)
        White_score = self.get_score(self.White)
            
        if Black_score == White_score:
            return 0 # 引き分け
        elif Black_score > White_score:
            return self.Black # Blackの勝ち
        elif Black_score < White_score:
            return self.White # Whiteの勝ち
        
    def get_score(self, color):
        """ 指定した色の現在のスコアを返す """
        score = 0
        for i in self.enable_actions:
            if self.get_cells(i) == color:
                score += 1
        return score


    def get_enables(self, color):
        result = []
        """ 置ける位置のリストを返す """
        for action in self.enable_actions:
            if self.get_cells(action) == self.Blank:
                """ 空白の位置 """
                if self.put_piece(action, color, False) > 0:
                    """ ここ置ける!! """
                    result.insert(0, action)
        return result
                    

    def update(self, action, color):
        """
        action:石を置く位置 0〜63
        """
        # そのマスにおいた場合の取れる数
        n = self.put_piece(action, color, False)

        if n  > 0:
            # そのマスは有効です
            self.put_piece(action, color)

            # 次のプレーヤーを登録する 
            if color == self.Black:                         #今、黒が終わって
                if len(self.get_enables(self.White))> 0:    #白の置けるマスが残っていたら
                    self.next_player = self.White           #次のプレーヤーは白

            else:                                           #今、白が終わって
                if len(self.get_enables(self.Black))> 0:    #黒の置けるマスが残っていたら
                    self.next_player = self.Black           #次のプレーヤーは黒

        return n
            

    def isEnd(self):
        e1 = self.get_enables(self.Black)        
        e2 = self.get_enables(self.White)  
        if len(e1) == 0 and len(e2) == 0:
            #双方置けなくなったらゲーム終了
            return True
            
        for action in self.enable_actions:
            if self.get_cells(action) == self.Blank:
                return False

        return True

 
if __name__ == "__main__":
   # game
    env = Reversi()
    print("------------- GAME START ---------------")
    while not env.isEnd():
        for i in range(1,3):
            if i == env.Black:
                print("*** 先手ターン● ***")
            else:
                print("*** 後手ターン○ ***")
            env.print_screen()
            enables = env.get_enables(i)
            if len(enables) > 0:
                flg = False
                while not flg:
                    print("番号を入力してください")
                    print(enables)
                    inp = input('>>>  ')
                    action_t = int(inp)
                    for j in enables:                
                        if action_t == j:
                            flg = True                       
                            break
                n = env.execute_action(action_t, i)

            else:
                print("パス")
                       

    print("*** ゲーム終了 ***")
    env.print_screen()
    if env.winner() == env.Black:
        print("先手●の勝ち！ スコアは、{:}/{:}です。".format(env.get_score(env.Black),len(env.enable_actions)))
    else:
        print("後手○の勝ち！ スコアは、{:}/{:}です。".format(env.get_score(env.White),len(env.enable_actions)))
 
