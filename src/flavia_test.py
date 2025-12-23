# import pandas as pd
# import pickle
# data = pd.read_pickle('/Users/flaviamicota/work/scamp-ml/schlaby-asp-gnn-3aprilie/data/instances/asp/config_ASP_TUBES_ORIGINAL_GNN_train-flavia.pkl')
# print(len(data))
#
# for instance in data[:2]:
#     for task in instance:
#         print("Task index", task.task_index, "quatity", task.quantity, task.machines)
#         # for machine_id in task.execution_times:
#         #     print("\tmachine", machine_id, "setup", task.setup_times[machine_id], "exectime", task.execution_times[machine_id], "execution_times_setup", task.execution_times_setup[machine_id]  )


gp_rez=[35204, 630735, 54753, 994335, 40854, 72097, 61463, 107760, 102135, 263427, 251443, 289151, 58516, 654846, 57090, 1089166, 117245, 174709, 54142, 149032, 888375, 344538, 64546, 351062, 228966, 45698, 229322, 66874, 471540, 116838, 664277, 768473, 81916, 98795, 313840, 88554, 408275, 154028, 946143, 101056, 251689, 620081, 141107, 48744, 126947, 47070, 511936, 757388, 108935, 117304, 57887, 46467, 65243, 934926, 35523, 1064338, 230520, 188791, 92215, 323584, 57555, 447572, 640488, 952943, 33017, 98689, 381609, 129053, 81205, 276851, 51577, 147955, 20034, 28133, 154397, 166892, 51153, 134676, 85705, 64367, 74870, 67441, 161357, 58463, 553942, 317745, 370418, 393507, 31156, 277535, 273037, 382436, 64991, 771858, 55927, 95264, 236296, 376334, 152067, 397306, 86265, 51778, 103531, 67545, 89300, 79146, 222800, 26433, 32059, 258311]
gp_prob= ['bom_tubes_p80_ao18_am18_4_m35_ao18_am30_2', 'bom_tubes_p80_ao18_am18_5_m35_ao18_am30_4', 'bom_tubes_p70_ao17_am18_4_m30_ao17_am25_1', 'bom_tubes_p80_ao18_am18_5', 'bom_tubes_p70_ao17_am18_3_m30_ao17_am23_2', 'bom_tubes_p80_ao18_am18_1_m35_ao18_am30_2', 'bom_tubes_p80_ao18_am18_5_m35_ao18_am28_2', 'bom_tubes_p80_ao18_am18_2_m35_ao18_am29_3', 'bom_tubes_p70_ao17_am18_1_m30_ao17_am28_5', 'bom_tubes_p80_ao18_am18_3_m35_ao18_am28_1', 'bom_tubes_p70_ao17_am18_2_m30_ao17_am23_4', 'bom_tubes_p70_ao17_am18_5_m10_ao17_am10_4', 'bom_tubes_p80_ao18_am18_4_m13_ao18_am13_3', 'bom_tubes_p70_ao17_am18_2_m10_ao17_am10_1', 'bom_tubes_p70_ao17_am18_4_m10_ao17_am10_2', 'bom_tubes_p80_ao18_am18_5_m13_ao18_am11_4', 'bom_tubes_p70_ao17_am18_1_m10_ao17_am10_3', 'bom_tubes_p80_ao18_am18_1_m13_ao18_am13_2', 'bom_tubes_p70_ao17_am18_4_m10_ao17_am10_3', 'bom_tubes_p80_ao18_am18_4_m13_ao18_am13_2', 'bom_tubes_p80_ao18_am18_1_m13_ao18_am11_3', 'bom_tubes_p80_ao18_am18_2_m13_ao18_am13_1', 'bom_tubes_p70_ao17_am18_5_m10_ao17_am10_5', 'bom_tubes_p70_ao17_am18_5_m30_ao17_am26_3', 'bom_tubes_p80_ao18_am18_3_m35_ao18_am29_4', 'bom_tubes_p80_ao18_am18_4', 'bom_tubes_p80_ao18_am18_2_m35_ao18_am31_4', 'bom_tubes_p70_ao17_am18_3_m30_ao17_am25_5', 'bom_tubes_p70_ao17_am18_5_m10_ao17_am10_2', 'bom_tubes_p70_ao17_am18_3_m10_ao17_am10_1', 'bom_tubes_p80_ao18_am18_3_m13_ao18_am12_4', 'bom_tubes_p80_ao18_am18_5_m13_ao18_am13_3', 'bom_tubes_p70_ao17_am18_4_m10_ao17_am10_4', 'bom_tubes_p70_ao17_am18_1', 'bom_tubes_p80_ao18_am18_1_m13_ao18_am13_5', 'bom_tubes_p70_ao17_am18_1_m10_ao17_am10_4', 'bom_tubes_p80_ao18_am18_3_m13_ao18_am11_1', 'bom_tubes_p70_ao17_am18_1_m30_ao17_am26_2', 'bom_tubes_p80_ao18_am18_5_m35_ao18_am27_1', 'bom_tubes_p70_ao17_am18_1_m30_ao17_am24_3', 'bom_tubes_p70_ao17_am18_2_m30_ao17_am26_1', 'bom_tubes_p80_ao18_am18_3', 'bom_tubes_p80_ao18_am18_2_m35_ao18_am28_1', 'bom_tubes_p80_ao18_am18_4_m35_ao18_am32_5', 'bom_tubes_p70_ao17_am18_2_m30_ao17_am27_5', 'bom_tubes_p80_ao18_am18_2_m35_ao18_am33_2', 'bom_tubes_p70_ao17_am18_2_m30_ao17_am23_2', 'bom_tubes_p80_ao18_am18_5_m35_ao18_am28_5', 'bom_tubes_p80_ao18_am18_1_m35_ao18_am30_5', 'bom_tubes_p80_ao18_am18_2', 'bom_tubes_p70_ao17_am18_3_m30_ao17_am25_3', 'bom_tubes_p80_ao18_am18_1_m35_ao18_am32_4', 'bom_tubes_p70_ao17_am18_1_m10_ao17_am10_5', 'bom_tubes_p80_ao18_am18_1_m13_ao18_am13_4', 'bom_tubes_p70_ao17_am18_4_m10_ao17_am10_5', 'bom_tubes_p80_ao18_am18_5_m13_ao18_am13_2', 'bom_tubes_p80_ao18_am18_3_m13_ao18_am12_5', 'bom_tubes_p80_ao18_am18_4_m13_ao18_am13_4', 'bom_tubes_p70_ao17_am18_5_m10_ao17_am10_3', 'bom_tubes_p80_ao18_am18_2_m13_ao18_am11_5', 'bom_tubes_p70_ao17_am18_3', 'bom_tubes_p80_ao18_am18_3_m13_ao18_am13_2', 'bom_tubes_p70_ao17_am18_2_m10_ao17_am10_5', 'bom_tubes_p80_ao18_am18_5_m13_ao18_am13_1', 'bom_tubes_p70_ao17_am18_3_m10_ao17_am10_3', 'bom_tubes_p80_ao18_am18_5_m13_ao18_am12_5', 'bom_tubes_p80_ao18_am18_2_m13_ao18_am13_4', 'bom_tubes_p80_ao18_am18_1_m35_ao18_am26_1', 'bom_tubes_p80_ao18_am18_4_m35_ao18_am29_4', 'bom_tubes_p80_ao18_am18_3_m35_ao18_am28_5', 'bom_tubes_p70_ao17_am18_4_m30_ao17_am27_4', 'bom_tubes_p80_ao18_am18_1', 'bom_tubes_p70_ao17_am18_4_m30_ao17_am25_5', 'bom_tubes_p70_ao17_am18_3_m30_ao17_am24_4', 'bom_tubes_p70_ao17_am18_1_m10_ao17_am9_2', 'bom_tubes_p70_ao17_am18_5_m30_ao17_am25_2', 'bom_tubes_p70_ao17_am18_1_m30_ao17_am26_1', 'bom_tubes_p70_ao17_am18_2_m30_ao17_am24_3', 'bom_tubes_p70_ao17_am18_3_m30_ao17_am25_1', 'bom_tubes_p80_ao18_am18_4_m35_ao18_am31_3', 'bom_tubes_p80_ao18_am18_3_m35_ao18_am30_2', 'bom_tubes_p80_ao18_am18_4_m35_ao18_am28_1', 'bom_tubes_p70_ao17_am18_5_m10_ao17_am10_1', 'bom_tubes_p70_ao17_am18_3_m10_ao17_am10_2', 'bom_tubes_p70_ao17_am18_2_m10_ao17_am10_4', 'bom_tubes_p80_ao18_am18_3_m13_ao18_am13_3', 'bom_tubes_p70_ao17_am18_2', 'bom_tubes_p80_ao18_am18_2_m35_ao18_am28_5', 'bom_tubes_p70_ao17_am18_4_m30_ao17_am25_3', 'bom_tubes_p70_ao17_am18_5_m30_ao17_am25_5', 'bom_tubes_p80_ao18_am18_1_m13_ao18_am13_1', 'bom_tubes_p70_ao17_am18_5', 'bom_tubes_p80_ao18_am18_4_m13_ao18_am12_5', 'bom_tubes_p70_ao17_am18_2_m10_ao17_am10_3', 'bom_tubes_p70_ao17_am18_3_m10_ao17_am10_5', 'bom_tubes_p80_ao18_am18_4_m13_ao18_am13_1', 'bom_tubes_p80_ao18_am18_2_m13_ao18_am13_2', 'bom_tubes_p80_ao18_am18_2_m13_ao18_am13_3', 'bom_tubes_p70_ao17_am18_3_m10_ao17_am10_4', 'bom_tubes_p70_ao17_am18_2_m10_ao17_am10_2', 'bom_tubes_p70_ao17_am18_4_m10_ao17_am10_1', 'bom_tubes_p70_ao17_am18_4', 'bom_tubes_p70_ao17_am18_1_m10_ao17_am10_1', 'bom_tubes_p70_ao17_am18_5_m30_ao17_am25_4', 'bom_tubes_p80_ao18_am18_5_m35_ao18_am31_3', 'bom_tubes_p80_ao18_am18_3_m35_ao18_am26_3', 'bom_tubes_p80_ao18_am18_1_m35_ao18_am29_3', 'bom_tubes_p70_ao17_am18_4_m30_ao17_am25_2', 'bom_tubes_p70_ao17_am18_1_m30_ao17_am23_4', 'bom_tubes_p70_ao17_am18_5_m30_ao17_am26_1']

poo_gnn_rez= [22721, 622294, 54753, 994335, 28503, 72097, 61463, 126009, 86647, 265240, 251443, 289513, 58516, 657327, 56008, 1049753, 106130, 176611, 45771, 85480, 888375, 327466, 70230, 348846, 227865, 45698, 179622, 50957, 476999, 110991, 664277, 768473, 74389, 98795, 333666, 86652, 407195, 154028, 890703, 101056, 251689, 620081, 141107, 48903, 126947, 43715, 516780, 757388, 110040, 117304, 57887, 46467, 65243, 914805, 30195, 1014239, 168147, 171062, 69807, 309009, 56730, 459007, 651941, 952943, 43820, 88758, 381907, 129053, 66305, 276851, 51577, 147955, 20034, 28133, 154397, 166892, 51153, 133876, 83017, 64367, 74870, 64770, 161357, 57556, 534332, 317745, 370418, 392949, 23606, 276627, 290568, 382436, 64649, 845171, 63969, 74976, 236296, 303802, 122628, 390242, 88871, 51778, 103531, 67545, 89300, 82715, 222800, 23479, 44275, 258311]

[64101, 753426, 140871, 1122300, 55070, 123546, 283601, 337033, 114034, 722297, 501744, 309907, 125502, 759927, 97704,
 1224789, 170589, 176611, 82912, 138033, 888475, 347635, 101987, 618171, 361671, 49175, 246929, 97441, 602495, 233732,
 1058613, 1628901, 169891, 99093, 314109, 98859, 462508, 199764, 1051235, 111726, 275487, 620181, 173615, 73172, 147334,
 65064, 674106, 826470, 244112, 129551, 129139, 52642, 75599, 988637, 75500, 1131265, 324869, 210521, 352440, 314448,
 63027, 652893, 661145, 1083367, 69434, 93420, 396133, 657449, 94616, 568984, 108592, 201897, 33329, 46132, 209732,
 232730, 106447, 341884, 116750, 96374, 272227, 84459, 164724, 92778, 669485, 446905, 571286, 507703, 36888, 316096,
 297728, 394610, 69008, 1006887, 102208, 136881, 368018, 398011, 158088, 631068, 116821, 60195, 148446, 84301, 109567,
 138987, 1236983, 56984, 129486, 510412]
letsa_min_prob=['bom_tubes_p70_ao17_am18_1',
'bom_tubes_p70_ao17_am18_1_m10_ao17_am9_2',
'bom_tubes_p70_ao17_am18_1_m10_ao17_am10_1',
'bom_tubes_p70_ao17_am18_1_m10_ao17_am10_3',
'bom_tubes_p70_ao17_am18_1_m10_ao17_am10_4',
'bom_tubes_p70_ao17_am18_1_m10_ao17_am10_5',
'bom_tubes_p70_ao17_am18_1_m30_ao17_am23_4',
'bom_tubes_p70_ao17_am18_1_m30_ao17_am24_3',
'bom_tubes_p70_ao17_am18_1_m30_ao17_am26_1',
'bom_tubes_p70_ao17_am18_1_m30_ao17_am26_2',
'bom_tubes_p70_ao17_am18_1_m30_ao17_am28_5',
'bom_tubes_p70_ao17_am18_2',
'bom_tubes_p70_ao17_am18_2_m10_ao17_am10_1',
'bom_tubes_p70_ao17_am18_2_m10_ao17_am10_2',
'bom_tubes_p70_ao17_am18_2_m10_ao17_am10_3',
'bom_tubes_p70_ao17_am18_2_m10_ao17_am10_4',
'bom_tubes_p70_ao17_am18_2_m10_ao17_am10_5',
'bom_tubes_p70_ao17_am18_2_m30_ao17_am23_2',
'bom_tubes_p70_ao17_am18_2_m30_ao17_am23_4',
'bom_tubes_p70_ao17_am18_2_m30_ao17_am24_3',
'bom_tubes_p70_ao17_am18_2_m30_ao17_am26_1',
'bom_tubes_p70_ao17_am18_2_m30_ao17_am27_5',
'bom_tubes_p70_ao17_am18_3',
'bom_tubes_p70_ao17_am18_3_m10_ao17_am10_1',
'bom_tubes_p70_ao17_am18_3_m10_ao17_am10_2',
'bom_tubes_p70_ao17_am18_3_m10_ao17_am10_3',
'bom_tubes_p70_ao17_am18_3_m10_ao17_am10_4',
'bom_tubes_p70_ao17_am18_3_m10_ao17_am10_5',
'bom_tubes_p70_ao17_am18_3_m30_ao17_am23_2',
'bom_tubes_p70_ao17_am18_3_m30_ao17_am24_4',
'bom_tubes_p70_ao17_am18_3_m30_ao17_am25_1',
'bom_tubes_p70_ao17_am18_3_m30_ao17_am25_3',
'bom_tubes_p70_ao17_am18_3_m30_ao17_am25_5',
'bom_tubes_p70_ao17_am18_4',
'bom_tubes_p70_ao17_am18_4_m10_ao17_am10_1',
'bom_tubes_p70_ao17_am18_4_m10_ao17_am10_2',
'bom_tubes_p70_ao17_am18_4_m10_ao17_am10_3',
'bom_tubes_p70_ao17_am18_4_m10_ao17_am10_4',
'bom_tubes_p70_ao17_am18_4_m10_ao17_am10_5',
'bom_tubes_p70_ao17_am18_4_m30_ao17_am25_1',
'bom_tubes_p70_ao17_am18_4_m30_ao17_am25_2',
'bom_tubes_p70_ao17_am18_4_m30_ao17_am25_3',
'bom_tubes_p70_ao17_am18_4_m30_ao17_am25_5',
'bom_tubes_p70_ao17_am18_4_m30_ao17_am27_4',
'bom_tubes_p70_ao17_am18_5',
'bom_tubes_p70_ao17_am18_5_m10_ao17_am10_1',
'bom_tubes_p70_ao17_am18_5_m10_ao17_am10_2',
'bom_tubes_p70_ao17_am18_5_m10_ao17_am10_3',
'bom_tubes_p70_ao17_am18_5_m10_ao17_am10_4',
'bom_tubes_p70_ao17_am18_5_m10_ao17_am10_5',
'bom_tubes_p70_ao17_am18_5_m30_ao17_am25_2',
'bom_tubes_p70_ao17_am18_5_m30_ao17_am25_4',
'bom_tubes_p70_ao17_am18_5_m30_ao17_am25_5',
'bom_tubes_p70_ao17_am18_5_m30_ao17_am26_1',
'bom_tubes_p70_ao17_am18_5_m30_ao17_am26_3',
'bom_tubes_p80_ao18_am18_1',
'bom_tubes_p80_ao18_am18_1_m13_ao18_am11_3',
'bom_tubes_p80_ao18_am18_1_m13_ao18_am13_1',
'bom_tubes_p80_ao18_am18_1_m13_ao18_am13_2',
'bom_tubes_p80_ao18_am18_1_m13_ao18_am13_4',
'bom_tubes_p80_ao18_am18_1_m13_ao18_am13_5',
'bom_tubes_p80_ao18_am18_1_m35_ao18_am26_1',
'bom_tubes_p80_ao18_am18_1_m35_ao18_am29_3',
'bom_tubes_p80_ao18_am18_1_m35_ao18_am30_2',
'bom_tubes_p80_ao18_am18_1_m35_ao18_am30_5',
'bom_tubes_p80_ao18_am18_1_m35_ao18_am32_4',
'bom_tubes_p80_ao18_am18_2',
'bom_tubes_p80_ao18_am18_2_m13_ao18_am11_5',
'bom_tubes_p80_ao18_am18_2_m13_ao18_am13_1',
'bom_tubes_p80_ao18_am18_2_m13_ao18_am13_2',
'bom_tubes_p80_ao18_am18_2_m13_ao18_am13_3',
'bom_tubes_p80_ao18_am18_2_m13_ao18_am13_4',
'bom_tubes_p80_ao18_am18_2_m35_ao18_am28_1',
'bom_tubes_p80_ao18_am18_2_m35_ao18_am28_5',
'bom_tubes_p80_ao18_am18_2_m35_ao18_am29_3',
'bom_tubes_p80_ao18_am18_2_m35_ao18_am31_4',
'bom_tubes_p80_ao18_am18_2_m35_ao18_am33_2',
'bom_tubes_p80_ao18_am18_3',
'bom_tubes_p80_ao18_am18_3_m13_ao18_am11_1',
'bom_tubes_p80_ao18_am18_3_m13_ao18_am12_4',
'bom_tubes_p80_ao18_am18_3_m13_ao18_am12_5',
'bom_tubes_p80_ao18_am18_3_m13_ao18_am13_2',
'bom_tubes_p80_ao18_am18_3_m13_ao18_am13_3',
'bom_tubes_p80_ao18_am18_3_m35_ao18_am26_3',
'bom_tubes_p80_ao18_am18_3_m35_ao18_am28_1',
'bom_tubes_p80_ao18_am18_3_m35_ao18_am28_5',
'bom_tubes_p80_ao18_am18_3_m35_ao18_am29_4',
'bom_tubes_p80_ao18_am18_3_m35_ao18_am30_2',
'bom_tubes_p80_ao18_am18_4',
'bom_tubes_p80_ao18_am18_4_m13_ao18_am12_5',
'bom_tubes_p80_ao18_am18_4_m13_ao18_am13_1',
'bom_tubes_p80_ao18_am18_4_m13_ao18_am13_2',
'bom_tubes_p80_ao18_am18_4_m13_ao18_am13_3',
'bom_tubes_p80_ao18_am18_4_m13_ao18_am13_4',
'bom_tubes_p80_ao18_am18_4_m35_ao18_am28_1',
'bom_tubes_p80_ao18_am18_4_m35_ao18_am29_4',
'bom_tubes_p80_ao18_am18_4_m35_ao18_am30_2',
'bom_tubes_p80_ao18_am18_4_m35_ao18_am31_3',
'bom_tubes_p80_ao18_am18_4_m35_ao18_am32_5',
'bom_tubes_p80_ao18_am18_5',
'bom_tubes_p80_ao18_am18_5_m13_ao18_am11_4',
'bom_tubes_p80_ao18_am18_5_m13_ao18_am12_5',
'bom_tubes_p80_ao18_am18_5_m13_ao18_am13_1',
'bom_tubes_p80_ao18_am18_5_m13_ao18_am13_2',
'bom_tubes_p80_ao18_am18_5_m13_ao18_am13_3',
'bom_tubes_p80_ao18_am18_5_m35_ao18_am27_1',
'bom_tubes_p80_ao18_am18_5_m35_ao18_am28_2',
'bom_tubes_p80_ao18_am18_5_m35_ao18_am28_5',
'bom_tubes_p80_ao18_am18_5_m35_ao18_am30_4',
'bom_tubes_p80_ao18_am18_5_m35_ao18_am31_3',]


for_b=[]

letsa_min_rez=[98795,
154397,
103531,
101913,
86652,
65243,
32059,
101056,
51153,
154028,
109499,
370418,
654846,
379761,
771858,
530026,
623477,
296883,
251443,
133876,
251689,
126947,
56730,
114097,
66447,
52266,
183042,
57131,
28503,
28133,
83017,
57887,
50957,
51778,
79496,
55082,
42320,
74389,
31987,
54753,
23479,
23606,
20034,
51577,
382436,
161357,
471540,
69807,
289151,
77992,
166892,
67545,
276627,
258311,
348846,
147955,
920998,
273037,
174709,
914805,
313840,
129053,
222800,
72097,
108935,
46467,
92499,
309009,
255381,
236296,
293209,
297764,
141107,
392949,
107760,
179622,
40854,
620081,
407195,
664277,
168147,
441099,
317745,
79146,
263427,
276851,
227865,
74870,
45698,
64635,
87257,
85480,
58516,
171062,
71044,
66305,
22721,
63997,
47424,
994335,
1049753,
88758,
952943,
1014239,
768473,
888890,
61463,
757388,
622294,
89300]


mop_min_rez=[98795,
154397,
103531,
105249,
88554,
65243,
32059,
101056,
51153,
154028,
86647,
370418,
654846,
379761,
771858,
533008,
651941,
511936,
251443,
133876,
251689,
126947,
56730,
113975,
58463,
32143,
134746,
55927,
38682,
28133,
85705,
57887,
63190,
51778,
86265,
56008,
50708,
81916,
31138,
54753,
24234,
24436,
20034,
51577,
382436,
161357,
471540,
90371,
289151,
64546,
166892,
67545,
277535,
258311,
348846,
147955,
888375,
273037,
174709,
934926,
313840,
129053,
222800,
72097,
108935,
46467,
117304,
309009,
327466,
236296,
376334,
381609,
141107,
392949,
107760,
229147,
41787,
620081,
407195,
664277,
191356,
441099,
317745,
79146,
263427,
276851,
228201,
74870,
45698,
62278,
84408,
130572,
58516,
176932,
67441,
66305,
25930,
63997,
47563,
994335,
1089166,
88758,
952943,
1021572,
768473,
943159,
61463,
757388,
630735,
89300]
equal = 0
gp_win = 0
gp_lose =0
i=0
for prob_l in letsa_min_prob:
    j=0
    for prob_g in gp_prob:
        if prob_l ==prob_g:
            break
        j+=1
    if mop_min_rez[i] == gp_rez[j]:#mop_min_rez,letsa_min_rez
        equal+=1
    elif mop_min_rez[i] < gp_rez[j]:
        gp_lose +=1
    else:
        gp_win +=1
    i +=1

print("gp vs mop equal", equal, 'win', gp_win, 'lose', gp_lose)

equal = 0
gp_win = 0
gp_lose =0
i=0
for prob_l in letsa_min_prob:
    j=0
    for prob_g in gp_prob:
        if prob_l ==prob_g:
            break
        j+=1
    if mop_min_rez[i] == poo_gnn_rez[j]:#mop_min_rez,letsa_min_rez
        equal+=1
    elif mop_min_rez[i] < poo_gnn_rez[j]:
        gp_lose +=1
    else:
        gp_win +=1
    i +=1

print("poo_gnn_rez vs mop equal", equal, 'win', gp_win, 'lose', gp_lose)



equal = 0
gp_win = 0
gp_lose =0
i=0
for prob_l in letsa_min_prob:
    j=0
    for prob_g in gp_prob:
        if prob_l ==prob_g:
            break
        j+=1
    if gp_rez[i] == poo_gnn_rez[j]:#mop_min_rez,letsa_min_rez
        equal+=1
    elif gp_rez[i] < poo_gnn_rez[j]:
        gp_lose +=1
    else:
        gp_win +=1
    i +=1

print("poo_gnn_rez vs gp", equal, 'win', gp_win, 'lose', gp_lose)



equal = 0
gp_win = 0
gp_lose =0
for i  in range(len(mop_min_rez)):
    if mop_min_rez[i] == letsa_min_rez[i]:
        equal += 1
    elif mop_min_rez[i] < letsa_min_rez[i]:
        gp_lose += 1
    else:
        gp_win += 1

print("letsa vs mop equal", equal, 'win', gp_win, 'lose', gp_lose)