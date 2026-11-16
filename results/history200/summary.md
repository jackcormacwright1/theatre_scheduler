| method                              |   setting |   patients_booked |   overrun_rate |   p90_list_risk |   overtime_per_list |   utilisation |   touch_utilisation |   long_case_share |   must_book_unbooked |
|:------------------------------------|----------:|------------------:|---------------:|----------------:|--------------------:|--------------:|--------------------:|------------------:|---------------------:|
| Book to average (queue order)       |      0.75 |              94.6 |            1.6 |             3.7 |                 0.7 |          71   |                59.3 |              27.6 |                 9.7  |
| Book to average (queue order)       |      0.8  |             101   |            3.3 |             7.3 |                 1.4 |          76   |                63.4 |              27.6 |                 6.9  |
| Book to average (queue order)       |      0.85 |             107.2 |            6.3 |            11.9 |                 2.6 |          80.5 |                66.9 |              27.3 |                 4.85 |
| Book to average (queue order)       |      0.9  |             113.6 |           12   |            19.3 |                 5   |          85.1 |                70.6 |              27.4 |                 3.3  |
| Book to average (queue order)       |      0.95 |             119.6 |           21.4 |            29.9 |                 9.4 |          89.2 |                73.8 |              27.4 |                 2.3  |
| Book to average (queue order)       |      1    |             125.4 |           34.6 |            46.1 |                16.5 |          92.7 |                76.6 |              27.9 |                 1.4  |
| Book to average (ratio order)       |      0.75 |              98.2 |            1.5 |             3.7 |                 0.6 |          70.9 |                58.6 |              25.4 |                 9.7  |
| Book to average (ratio order)       |      0.8  |             106.2 |            3   |             7   |                 1.3 |          75.8 |                62.3 |              24.3 |                 6.9  |
| Book to average (ratio order)       |      0.85 |             114.2 |            5.6 |            11.3 |                 2.3 |          80.3 |                65.6 |              23.1 |                 4.85 |
| Book to average (ratio order)       |      0.9  |             122.8 |           10.8 |            18.9 |                 4.4 |          85   |                69   |              21.9 |                 3.3  |
| Book to average (ratio order)       |      0.95 |             131.5 |           19.8 |            29.5 |                 8.3 |          89.3 |                72   |              20.7 |                 2.3  |
| Book to average (ratio order)       |      1    |             139   |           32.8 |            45.8 |                14.5 |          92.8 |                74.6 |              20   |                 1.4  |
| Chance-constrained (true durations) |      0.02 |             100.9 |            1.3 |             2.1 |                 0.4 |          71.9 |                59.3 |              24.4 |                10.45 |
| Chance-constrained (true durations) |      0.05 |             109.1 |            3.2 |             4.8 |                 1.1 |          77.2 |                63.3 |              23.5 |                 7.55 |
| Chance-constrained (true durations) |      0.1  |             116.6 |            6.7 |             9.6 |                 2.3 |          81.8 |                66.8 |              22.6 |                 5.3  |
| Chance-constrained (true durations) |      0.2  |             125   |           13.9 |            19.4 |                 5.1 |          86.7 |                70.4 |              21.8 |                 3.5  |
| Chance-constrained (true durations) |      0.3  |             131.8 |           22.3 |            29.3 |                 8.8 |          90.1 |                72.8 |              21   |                 2.45 |
| Chance-constrained (plug-in model)  |      0.02 |             100.6 |            1.4 |             2.5 |                 0.5 |          71.8 |                59.2 |              24.7 |                10.5  |
| Chance-constrained (plug-in model)  |      0.05 |             109.3 |            3.5 |             5.4 |                 1.2 |          77.5 |                63.5 |              23.7 |                 7.1  |
| Chance-constrained (plug-in model)  |      0.1  |             116.8 |            7.1 |            10.9 |                 2.5 |          82.1 |                67   |              22.7 |                 5.1  |
| Chance-constrained (plug-in model)  |      0.2  |             125.3 |           14.4 |            21   |                 5.4 |          86.8 |                70.5 |              21.9 |                 3.5  |
| Chance-constrained (plug-in model)  |      0.3  |             131.7 |           22.6 |            30.9 |                 9   |          90.1 |                72.8 |              21   |                 2.25 |

Extra patients vs book-to-average at the same realised overrun rate

| method    | vs        |   setting |   worlds |   extra_patients |   se |
|:----------|:----------|----------:|---------:|-----------------:|-----:|
| cc-plugin | avg-queue |      0.02 |       10 |             8.33 | 1.64 |
| cc-plugin | avg-queue |      0.05 |       20 |             8.02 | 1.24 |
| cc-plugin | avg-queue |      0.1  |       20 |             8.81 | 1.15 |
| cc-plugin | avg-queue |      0.2  |       20 |            10.16 | 1.15 |
| cc-plugin | avg-queue |      0.3  |       20 |            11.5  | 1.12 |
| cc-oracle | avg-queue |      0.02 |        7 |             9.23 | 1.28 |
| cc-oracle | avg-queue |      0.05 |       19 |             8.63 | 1.1  |
| cc-oracle | avg-queue |      0.1  |       20 |             9.16 | 1.18 |
| cc-oracle | avg-queue |      0.2  |       20 |            10.33 | 1.1  |
| cc-oracle | avg-queue |      0.3  |       20 |            11.72 | 1.14 |
| cc-plugin | avg-ratio |      0.02 |       10 |             1.51 | 0.42 |
| cc-plugin | avg-ratio |      0.05 |       20 |             1.3  | 0.43 |
| cc-plugin | avg-ratio |      0.1  |       20 |             0.02 | 0.21 |
| cc-plugin | avg-ratio |      0.2  |       20 |            -1.15 | 0.21 |
| cc-plugin | avg-ratio |      0.3  |       20 |            -1.42 | 0.25 |
| cc-oracle | avg-ratio |      0.02 |        9 |             3.28 | 0.55 |
| cc-oracle | avg-ratio |      0.05 |       19 |             1.69 | 0.35 |
| cc-oracle | avg-ratio |      0.1  |       20 |             0.55 | 0.28 |
| cc-oracle | avg-ratio |      0.2  |       20 |            -0.83 | 0.24 |
| cc-oracle | avg-ratio |      0.3  |       20 |            -1.2  | 0.2  |
