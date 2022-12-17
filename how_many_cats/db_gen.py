import numpy as np
import pandas as pd
import re



if __name__ == '__main__':
    cat_life_expectancy = 8
    cat_life_expectancy_std = 3
    num_cats = 3
    kings_life_expectancy = 100
    cats_names = ['Oreo',
                  'Milo',
                  'Leo',
                  'Hyde',
                  'Tito',
                  'Sasha',
                  'Kiwi',
                  'Athena',
                  'Brownie',
                  'Birch',
                  'Rose',
                  'Angel',
                  'Ozzy',
                  'Ferris',
                  'Hercules',
                  'Summer',
                  'Shakira',
                  'Honey',
                  'Mars',
                  'Smudge',
                  'Dolly',
                  'Dandelion',
                  'Patches',
                  'Stormi',
                  'Truffle',
                  'Bluebell',
                  'Petunia',
                  'Nilla',
                  'Bear',
                  'Jade',
                  'DojaCat',
                  'Yoda',
                  'Mango',
                  'Venus',
                  'Maple',
                  'Felix',
                  'Puss',
                  'Lola',
                  'Buffy',
                  'Meatball',
                  'KitKat',
                  'Lucky',
                  'Pepper',
                  'Miso',
                  'Mamba',
                  'Binx',
                  'Cheerio',
                  'Whoopi',
                  'Cappuccino',
                  'Penny',
                  'Vera',
                  'Apollo',
                  'Titan',
                  'Bandit',
                  'Ginko',
                  'Cookie',
                  'Ruby',
                  'Aries',
                  'Pearl',
                  'Buckeye',
                  'Java',
                  'Snowflake',
                  'Butterscotch',
                  'Tigger',
                  'Lady',
                  'Cher',
                  'Elsa',
                  'Delia',
                  'Catzilla',
                  'Oscar',
                  'Blossom',
                  'Topaz',
                  'Weasel',
                  'Luna',
                  'Fluffy',
                  'Godiva',
                  'Simba',
                  'Buttercup',
                  'Clementine',
                  'Waffle',
                  'Whiskers',
                  'Fern',
                  'Smelly',
                  'Pickle',
                  'Moose',
                  'Badger',
                  'Winston',
                  'Rain',
                  'Mocha',
                  'River',
                  'Midnight',
                  'Garfield',
                  'Kanye',
                  'Pip',
                  'Wasabi',
                  'Bruno',
                  'Agate',
                  'Princess',
                  'Whiskey',
                  'Coco',
                  'Cheddar',
                  'Duke',
                  'Nyx',
                  'Chestnut',
                  'Jameson',
                  'Daisy',
                  'Sushi',
                  'Mowgli',
                  'Sylvester',
                  'Opal',
                  'Ivy',
                  'Mila',
                  'Mifflin',
                  'LordTubbington',
                  'Crookshanks',
                  'Medusa',
                  'Mittens',
                  'Dash',
                  'Willow',
                  'Everest',
                  'Zazzles',
                  'Malcolm',
                  'Ginger',
                  'Hades',
                  'Olive',
                  'Rajah',
                  'Clover',
                  'Oliver',
                  'Chip',
                  'Regina',
                  'Onyx',
                  'Socks',
                  'Atticus',
                  'Scratchy',
                  'Beckham',
                  'Mochi',
                  'Aslan',
                  'Nala',
                  'Elvis',
                  'Lucifer',
                  'Salem',
                  'Jesse',
                  'Beatrix',
                  'Toulouse',
                  'Ajax',
                  'Bella',
                  'Gaga',
                  'Cheeto',
                  'Gatsby',
                  'Loki',
                  'Orchid',
                  'Remus',
                  'Oprah',
                  'Usher',
                  'Zeus',
                  'Max',
                  'Peanut',
                  'Mufasa']

    cats_life = 1 + np.random.normal(cat_life_expectancy, cat_life_expectancy_std, num_cats).astype(int)
    cats_life = np.clip(cats_life, a_min=1, a_max=9999999)
    cats_birth_year = np.random.randint(0, kings_life_expectancy, num_cats).astype(int)
    cats_death_year = cats_birth_year + cats_life

    mx_death_year = cats_death_year.max()
    df = pd.DataFrame({'birth_year': cats_birth_year, 'death_year': cats_death_year})
    df['age'] = df['death_year'] - df['birth_year']
    df['name'] = np.random.choice(cats_names, num_cats, replace=False)

    crondf = pd.DataFrame(index=range(mx_death_year), columns=['num_cats'], data=0)
    for idx, row in df.iterrows():
        crondf.loc[row['birth_year']:(row['death_year'] - 1), 'num_cats'] += 1
        j = 3

right_answer = 1 # np.random.choice(crondf['num_cats'].unique())
random_year = crondf[crondf['num_cats'].eq(right_answer)].sample(1).index.to_list()[0]
df = df.sample(len(df))

print("<><><><><><><><><><><><><><>")
print('INPUT:')
print(f'{num_cats}')
print(f'{random_year}')
for idx, row in df.iterrows():
    print(f'{row["name"]} {row["birth_year"]} {row["death_year"]}')
print("<><><><><><><><><><><><><><>")
print('OUTPUT:')
print(f'{right_answer}')

df = df.sort_values(by=['birth_year', 'death_year'])
j = 3
