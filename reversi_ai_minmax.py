# coding: utf8
import random
import time
import math
import copy
import tkinter
from tkinter import messagebox

# =========================
# リバーシ
# =========================

# ------------------------
# 定数（として扱う変数）
# ------------------------
SPACE = 0  # 空き
BLACK = 1  # 黒石
WHITE = -1  # 白石

BOARD_PX_SIZE = 500          # 盤面のサイズ
CELL_PX_SIZE = BOARD_PX_SIZE / 8  # マスのサイズ

AI_LEVEL = 3  # AIが読む深さ

# ------------------------
# マスの座標を管理するクラス
# ------------------------
class Position:
  def __init__(self):
    self.y = 0
    self.x = 0
  
  def __init__(self, y, x):
    self.y = y
    self.x = x
  
# ------------------------
# 盤面クラス
# ------------------------
class Board:
  # 8方向のYXの加算値
  DIR = [[-1, -1], [-1, 0], [-1, 1],
         [ 0, -1],          [ 0, 1],
         [ 1, -1], [ 1, 0], [ 1, 1]]  
  
  def __init__(self):
    # 盤面。8×8の2次元リストを生成
    self.board = \
      [[SPACE for i in range(8)] for j in range(8)]
    self.turn = BLACK  # 手番
    self.move_num = 1  # 手数
  
  # 盤面の初期化（初期配置）
  def init_board(self):
    for y in range(8):
      for x in range(8):
        self.board[y][x] = SPACE
    self.board[3][3] = WHITE
    self.board[3][4] = BLACK
    self.board[4][3] = BLACK
    self.board[4][4] = WHITE
    
    self.turn = BLACK  # 手番
    self.move_num = 1  # 手数
  
  # 黒白の石数をタプルで返す
  def get_discs(self):
    black_discs = 0
    white_discs = 0
    for y in range(8):
      for x in range(8):
        disc = self.board[y][x]
        if disc == BLACK:
          black_discs += 1
        elif disc == WHITE:
          white_discs += 1
    return (black_discs, white_discs)
  
  # 指定のマスに石を打てるか？
  def is_movable(self, position):
    # 空きでなければ打てない
    if self.board[position.y][position.x] != SPACE:
      return False
    
    # 各方向に石をひっくり返せるか？
    for dir in Board.DIR:
      y = position.y + dir[0]
      x = position.x + dir[1]
      if y >= 0 and x >= 0 and y < 8 and x < 8 \
        and self.board[y][x] == -self.turn:
        # 隣が相手の石
        y += dir[0]
        x += dir[1]
        while y >= 0 and x >= 0 and y < 8 and \
          x < 8 and self.board[y][x] == -self.turn:
          y += dir[0]
          x += dir[1]
        if y >= 0 and x >= 0 and y < 8 and x < 8 \
          and self.board[y][x] == self.turn:
          return True
      
    return False
  
  # 石を打てるマスのリストを返す
  def get_move_list(self):
    move_list = []
    for y in range(8):
      for x in range(8):
        if self.board[y][x] == SPACE:
          position = Position(y, x)
          if self.is_movable(position):
            move_list.append(position)
    return move_list
  
  # 局面を進める
  def move(self, position):
    # 石を打つ
    self.board[position.y][position.x] = self.turn
    
    # 石をひっくり返す
    # 各方向に石をひっくり返せるか調べる
    for dir in Board.DIR:
      y = position.y + dir[0]
      x = position.x + dir[1]
      if y >= 0 and x >= 0 and y < 8 and x < 8 \
        and self.board[y][x] == -self.turn:
        # 隣が相手の石
        y += dir[0]
        x += dir[1]
        while y >= 0 and x >= 0 and y < 8 and \
          x < 8 and self.board[y][x] == -self.turn:
          y += dir[0]
          x += dir[1]
        if y >= 0 and x >= 0 and y < 8 and x < 8 \
          and self.board[y][x] == self.turn:
          # この方向は返せる
          # 1マス戻る
          y -= dir[0]
          x -= dir[1]
          # 戻りながら返す
          while y >= 0 and x >= 0 and y < 8 and \
            x < 8 and self.board[y][x]==-self.turn:
            self.board[y][x] = self.turn
            y -= dir[0]
            x -= dir[1]
    
    self.turn = -self.turn  # 手番を変更
    self.move_num += 1    # 手数を増やす
  
  # パスする
  def move_pass(self):
    self.turn = -self.turn  # パス
  
  # 対局終了の判定
  def is_game_end(self):
    # 60手に達した時
    if self.move_num == 61:
      return True
    
    # 黒白どちらかの石数が0になった時
    # 黒白の石数をタプルで取得
    (black_discs, white_discs) = self.get_discs()
    if black_discs == 0 or white_discs == 0:
      return True
    
    # 黒白どちらも手がない場合
    move_list1 = self.get_move_list()
    if len(move_list1) == 0:
      self.move_pass()  # パスして相手番にする
      move_list2 = self.get_move_list()
      self.move_pass()  # パスして戻す
      if len(move_list2) == 0:
        return True
    
    return False
  
  # 局面評価
  # 黒:プラス  白:マイナス
  def evaluation(self):
    eval_val = 0
    
    # 終了局面であれば黒石数 - 白石数を返す
    if self.is_game_end():
      (black_discs, white_discs) = self.get_discs() # 黒白の石数をタプルで取得
      eval_val = black_discs - white_discs
      if eval_val > 0:
        eval_val += 1000
      elif eval_val < 0:
        eval_val -= 1000
      return eval_val
    
    # 隅
    corner = 0
    for (y, x) in [(0, 0), (0, 7), (7, 0), (7, 7)]:
      if self.board[y][x] == BLACK:
        corner += 1
      elif self.board[y][x] == WHITE:
        corner -= 1
    eval_val += corner * 16
    
    # 手番 打てるマスの数
    move_num = 0
    move_list = self.get_move_list()
    if self.turn == BLACK:
      move_num += len(move_list)
    else:
      move_num -= len(move_list)
    
    # 相手 打てるマスの数
    self.move_pass()
    move_list = self.get_move_list()
    if self.turn == BLACK:
      move_num += len(move_list)
    else:
      move_num -= len(move_list)
    self.move_pass()
    
    eval_val += move_num * 2
    
    return eval_val
    
# ------------------------
# ゲームクラス
# ------------------------
class Game:
  def __init__(self):
    # ゲームの状態 0:開始待ち 1:対局中 2:対局終了
    self.game_mode = 0
    # ● 0:プレイヤー 1:コンピュータ
    self.black_player = 0
    # ○ 0:プレイヤー 1:コンピュータ
    self.white_player = 0  
    self.board = Board()  # 盤面作成
    self.board.init_board()  # 盤面の初期化
  
  # 対局開始
  def start(self, _black_player, _white_player):
    self.black_player = _black_player
    self.white_player = _white_player
    self.game_mode = 1  # ゲームの状態:対局中
    self.board.init_board()  # 盤面の初期化
  
  # 局面を進める
  def game_move(self, position):
    self.board.move(position) # 局面を進める
    draw_board()        # 盤面を描画
    
    # 終局判定
    if self.board.is_game_end():
      self.game_mode = 2    # 対局終了
      disp_mess()        # メッセージ表示
      messagebox.showinfo(u"", u"対局終了")
      return
    
    # パス判定
    move_list = self.board.get_move_list()
    if len(move_list) == 0:
      # 石を打てる場所がないのでパス
      self.board.move_pass()  
      messagebox.showinfo(u"パス", \
        u"打てる場所がないのでパスします")
    
    disp_mess()          # メッセージ表示
    
  # 次の手番はコンピュータか？
  def is_com_turn(self):
    if self.board.turn == BLACK and \
      self.black_player == 1 or \
      self.board.turn == WHITE and \
      self.white_player == 1:
      return True
    return False
  
  # 次の手番がコンピュータなら
  # AIに指し手を選択させる
  def proc_com_turn(self):
    while True:
      if self.is_com_turn():
        position = AI().select_move(self.board)
        self.game_move(position) # 局面を進める
        if self.game_mode == 2:
          break  # 対局終了していたら抜ける
      else:
        break

# ------------------------
# AIクラス
# ------------------------
class AI:
  # 与えられた盤面から指し手を返す
  def select_move(self, board):
    # すべての指し手を生成
    move_list = board.get_move_list()
    # 同じ手になりにくくするためシャッフルする
    random.shuffle(move_list)
    
    ai_level = AI_LEVEL # 読みの深さ
    
    if board.turn == BLACK:
      best_eval = -10000  # 黒の手番:すごく低い評価値からスタート
    else:
      best_eval = 10000   # 白の手番:すごく高い評価値からスタート
    
    for position in move_list:
      tmp_board = copy.deepcopy(board)
      tmp_board.move(position)  # 局面を進める
      eval = self.minmax(tmp_board, ai_level - 1) # 局面を評価
      if board.turn == BLACK:
        if eval > best_eval: 
          # 黒の手番 : 評価値がより高い手の場合は最善手を入れ替える
          best_eval = eval
          best_position = position
      else:
        if eval < best_eval:
          # 白の手番 : 評価値がより低い手の場合は最善手を入れ替える
          best_eval = eval
          best_position = position
    
    print(str(board.move_num) + "手 best_eval: " + str(best_eval))
    return best_position
  
  def minmax(self, board, depth):
    # 引数で指定された深さまで読んだか終局となった場合は、局面評価値を返す
    if depth <= 0 or board.is_game_end():
      return board.evaluation()
    
    # すべての指し手を生成
    move_list = board.get_move_list()
    
    if len(move_list) == 0:
      # 打てる場所がないのでパス
      tmp_board = copy.deepcopy(board)
      tmp_board.move_pass()
      return self.minmax(tmp_board, depth)
    
    if board.turn == BLACK:
      best_eval = -10000  # 黒の手番:すごく低い評価値からスタート
    else:
      best_eval = 10000   # 白の手番:すごく高い評価値からスタート
    for position in move_list:
      tmp_board = copy.deepcopy(board)
      tmp_board.move(position)
      eval = self.minmax(tmp_board, depth - 1)
      if board.turn == BLACK:
        if eval > best_eval:  # 黒の手番 : 評価値がより高い手
          best_eval = eval
      else:
        if eval < best_eval:  # 白の手番 : 評価値がより低い手
          best_eval = eval
    
    return best_eval
  
# ------------------------
# UI関数
# ------------------------
# 盤面の描画
def draw_board():
  global game
  global canvas_board
  
  canvas_board.delete('all')  # キャンバスをクリア
  # 背景
  canvas_board.create_rectangle(0, 0, \
    BOARD_PX_SIZE, BOARD_PX_SIZE, fill = '#00a000')
  for y in range(8):
    for x in range(8):
      disc = game.board.board[y][x]
      if disc != SPACE:
        if disc == BLACK:
          color = "black"
        else:
          color = "white"
        # 石の描画
        canvas_board.create_oval( \
          x*CELL_PX_SIZE+4, y*CELL_PX_SIZE+4, \
          (x+1)*CELL_PX_SIZE-4, \
          (y+1)*CELL_PX_SIZE-4, fill=color)  
    
  # 枠を描画
  for x in range(8):
    canvas_board.create_line(x * CELL_PX_SIZE, \
      0, x * CELL_PX_SIZE, BOARD_PX_SIZE, \
      fill="black", width=1)
  for y in range(8):
    canvas_board.create_line(0, y * CELL_PX_SIZE, \
      BOARD_PX_SIZE, y * CELL_PX_SIZE, \
      fill="black", width=1)
    
  canvas_board.update()

# メッセージ表示
def disp_mess():
  global game
  global mess_var
  
  mess = ""
  if game.game_mode == 0:
    mess = u"対局を開始してください"
    
  elif game.game_mode == 1:
    mess = u"対局中 "
    mess += str(game.board.move_num) + u"手目 "
    if game.board.turn == BLACK:
      mess += u"黒番"
    else:
      mess += u"白番"
    # 黒白の石数をタプルで取得
    (black_discs, white_discs) = \
      game.board.get_discs()
    mess += " 黒:" + str(black_discs) + " 白:" \
      + str(white_discs)
    
  elif game.game_mode == 2:
    # 黒白の石数をタプルで取得
    (black_discs, white_discs) = \
      game.board.get_discs()  
    mess = u"対局終了 " + \
      str(game.board.move_num-1) + u"手" + " 黒:" \
      +str(black_discs)+" 白:"+str(white_discs)
    if black_discs == white_discs:
      mess += u" 引き分け"
    elif black_discs > white_discs:
      mess += u" 黒の勝ち "
    else:
      mess += u" 白の勝ち "
  mess_var.set(mess)  # メッセージラベルにセット

# 「対局開始」ボタンが押された時
def play_start():
  global game
  global black_var, white_var

  # ● 0:プレイヤー 1:コンピュータ
  black_player = black_var.get()
  # ○ 0:プレイヤー 1:コンピュータ
  white_player = white_var.get()  

  # 対局開始
  game.start(black_player, white_player)  
  disp_mess()     # メッセージ表示
  draw_board()    # 盤面を描画
  
  # 次の手番がコンピュータの場合
  # （プレイヤーの手番なら何もしない）
  game.proc_com_turn()

# 盤面がクリックされた時
def click_board(event):
  global game
  if game.game_mode != 1:
    messagebox.showinfo(u"", \
      u"対局開始してください")
    return
  y = math.floor(event.y / CELL_PX_SIZE)
  x = math.floor(event.x / CELL_PX_SIZE)
  position = Position(y, x)
  if game.board.is_movable(position) == False:
    messagebox.showinfo(u"", u"そこには打てません")
    return
  
  game.game_move(position)  # 局面を進める
  if game.game_mode == 2:
    return  # 対局終了していたら抜ける
  
  # 次の手番がコンピュータの場合
  # （プレイヤーの手番なら何もしない）
  game.proc_com_turn()

# ------------------------
# メイン処理
# ------------------------

# ウインドウ初期化
root = tkinter.Tk()
root.title(u"リバーシ")

# ウインドウの幅
window_width  = BOARD_PX_SIZE + 32
# ウインドウの高さ 
window_height = BOARD_PX_SIZE + 88
# ウインドウサイズを指定
root.geometry(str(window_width) + "x" + \
  str(window_height))   

# 盤面キャンバスを作成
# キャンバスを作成
canvas_board = tkinter.Canvas(root, \
  width = BOARD_PX_SIZE, height = BOARD_PX_SIZE)
# キャンバスがクリックされた時に呼び出す関数を設定
canvas_board.bind("<Button-1>", click_board)
# キャンバスの位置を指定  
canvas_board.place(x = 16, y = 72)        

# 対局条件
black_label = tkinter.Label(text=u"先手●")
black_label.place(x = 16, y = 4)
black_var = tkinter.IntVar()
black_rdo0 = tkinter.Radiobutton(root, value = 0, \
  variable = black_var, text = u"プレイヤー")
black_rdo0.place(x = 70, y = 4)
black_rdo1 = tkinter.Radiobutton(root, value = 1, \
  variable = black_var, text = u"コンピュータ")
black_rdo1.place(x = 160, y = 4)

white_label = tkinter.Label(text=u"後手○")
white_label.place(x = 16, y = 24)
white_var = tkinter.IntVar()
white_rdo0 = tkinter.Radiobutton(root, value = 0, \
  variable = white_var, text = u"プレイヤー")
white_rdo0.place(x = 70, y = 24)
white_rdo1 = tkinter.Radiobutton(root, value = 1, \
  variable = white_var, text = u"コンピュータ")
white_rdo1.place(x = 160, y = 24)

# 「対局開始」ボタンを設置
button_start = tkinter.Button(root, \
  text = u"対局開始", width=15, command=play_start)
button_start.place(x = 300, y = 12)

# メッセージ欄
mess_var = tkinter.StringVar()
mess_label = tkinter.Label(root, \
  textvariable = mess_var)
mess_label.place(x = 16, y = 48)

game = Game()   # ゲームインスタンス作成
draw_board()    # 盤面を描画
disp_mess()     # メッセージ表示

# GUIの待ち受けループ
root.mainloop()

