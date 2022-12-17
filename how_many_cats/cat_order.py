import pandas as pd

s = '''
100
89
Cappuccino 79 89
Aslan 39 50
Cheeto 13 21
Catzilla 16 18
Nyx 56 60
Petunia 60 69
Patches 47 57
Peanut 35 42
Penny 0 14
Waffle 87 94
Dolly 61 68
Lola 45 52
Lady 21 23
Midnight 65 70
Zazzles 41 49
Truffle 52 66
Coco 32 41
Ginger 79 87
Beatrix 34 40
Simba 70 76
Buttercup 76 85
Yoda 17 22
DojaCat 9 19
Butterscotch 11 20
Clover 71 84
Regina 74 80
Tigger 82 93
Mochi 0 3
Garfield 7 19
Binx 9 18
Angel 16 24
Aries 3 10
Rajah 57 66
Dash 69 74
Puss 76 80
Onyx 42 47
Shakira 8 16
Oprah 30 36
Rose 89 96
KitKat 23 35
Miso 33 37
Mango 22 33
Bandit 48 58
Kiwi 52 66
Rain 0 6
Whiskey 11 18
Leo 34 42
Salem 13 21
Agate 25 33
Malcolm 27 33
Clementine 84 91
Bear 64 69
Pearl 84 90
Opal 42 51
Summer 69 80
River 42 50
Chestnut 84 90
Socks 27 33
Elvis 67 81
Jade 59 67
Jameson 89 101
Oliver 63 71
Moose 1 12
Loki 64 70
Bluebell 42 51
Remus 89 99
Winston 52 58
Lucifer 65 76
Bella 86 93
Duke 64 69
Fluffy 86 91
Athena 76 85
Pip 10 12
Olive 10 15
Bruno 31 38
LordTubbington 29 38
Daisy 46 57
Pepper 53 62
Ozzy 68 76
Oreo 86 93
Nilla 1 16
Usher 7 12
Badger 41 53
Gaga 77 87
Everest 37 43
Dandelion 37 48
Princess 31 45
Mamba 77 84
Elsa 16 25
Buckeye 13 23
Ivy 78 81
Snowflake 53 61
Toulouse 87 93
Weasel 83 91
Cheerio 3 13
Luna 60 64
Smelly 68 76
Cookie 75 82
Milo 30 35
Stormi 56 64
'''

if __name__ == '__main__':
    ss = [st for st in s.split('\n') if st != '']
    n = int(ss[0])
    m = int(ss[1])
    cats = [st.split(' ') for st in ss[2:]]
    catsdf = pd.DataFrame(cats, columns=['name', 'birth year', 'death year'])
    catsdf['birth year'] = catsdf['birth year'].astype(int)
    catsdf['death year'] = catsdf['death year'].astype(int)
    catsdf = catsdf.sort_values(by=['name'])

    print(n)
    print(m)
    for idx, row in catsdf.iterrows():
        print(f'{row["name"]} {row["birth year"]} {row["death year"]}')
