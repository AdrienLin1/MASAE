from visualdl.server import app

list1 = [
         './HalfCheetah-v3_1_8', './HalfCheetah-v3_2_1',
         './HalfCheetah-v3_2_2', './HalfCheetah-v3_3_1',
         './HalfCheetah-v3_3_2', './HalfCheetah-v3_4_1',
         './HalfCheetah-v3_4_1',
         './HalfCheetah-v3_5_1', './HalfCheetah-v3_5_2',
         './HalfCheetah-v3_6_1', './HalfCheetah-v3_6_2',
         './HalfCheetah-v3_7_1', './HalfCheetah-v3_7_2',
         './HalfCheetah-v3_8_1',  './HalfCheetah-v3_9_1',
         './HalfCheetah-v3_10_1',  './HalfCheetah-v3_11_1',
         './HalfCheetah-v3_12_1',  './HalfCheetah-v3_13_1',
         './HalfCheetah-v3_13_2',
         './HalfCheetah-v3_14_1',  './HalfCheetah-v3_14_2',
         './HalfCheetah-v3_15_1', './HalfCheetah-v3_15_2',
         '/home/hkayi/PycharmProjects/maddpg_fa/HalfCheetah-v3_1_1',
         '/home/hkayi/PycharmProjects/maddpg_fa/HalfCheetah-v3_2_1',
         '/home/hkayi/PycharmProjects/maddpg_fa/HalfCheetah-v3_3_1',
         '/home/hkayi/PycharmProjects/maddpg_fa/HalfCheetah-v3_4_1',
         '/home/hkayi/PycharmProjects/maddpg_fa/HalfCheetah-v3_5_1',
         '/home/hkayi/PycharmProjects/maddpg_fa/HalfCheetah-v3_6_1',
         '/home/hkayi/PycharmProjects/maddpg_fa/HalfCheetah-v3_7_1',
         '/home/hkayi/PycharmProjects/maddpg_fa/HalfCheetah-v3_8_1',
         '/home/hkayi/PycharmProjects/maddpg_fa/HalfCheetah-v3_9_1', '/home/hkayi/PycharmProjects/maddpg_fa/HalfCheetah-v3_10_1',
         '/home/hkayi/PycharmProjects/maddpg_fa/HalfCheetah-v3_11_1', '/home/hkayi/PycharmProjects/maddpg_fa/HalfCheetah-v3_12_1',
         '/home/hkayi/PycharmProjects/maddpg_fa/HalfCheetah-v3_13_1', '/home/hkayi/PycharmProjects/maddpg_fa/HalfCheetah-v3_14_1',
         '/home/hkayi/PycharmProjects/maddpg_fa/HalfCheetah-v3_15_1'
         ]



if __name__ == '__main__':
    app.run(logdir=list1)
