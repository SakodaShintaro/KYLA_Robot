import sqlite3
import pprint

dbname = 'FACE_FEATURES.db'
conn = sqlite3.connect(dbname)
cur = conn.cursor()

# terminalで実行したSQL文と同じようにexecute()に書く
cur.execute('SELECT * FROM persons')
hoge = cur.fetchall()
# print(hoge)
print(hoge[3])
exit()

# 中身を全て取得するfetchall()を使って、printする。
# print(cur.fetchall())
pprint.pprint(cur.fetchall())

cur.close()
conn.close()