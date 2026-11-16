| method                              |   setting |   patients_booked |   overrun_rate |   p90_list_risk |   overtime_per_list |   utilisation |   touch_utilisation |   long_case_share |   must_book_unbooked |
|:------------------------------------|----------:|------------------:|---------------:|----------------:|--------------------:|--------------:|--------------------:|------------------:|---------------------:|
| Book to average (queue order)       |      0.75 |              94.2 |            1.7 |             4.2 |                 0.7 |          70.7 |                59.1 |              27.5 |                10.25 |
| Book to average (queue order)       |      0.8  |             100.1 |            3.8 |             9.1 |                 1.6 |          75.5 |                63.1 |              27.9 |                 7.35 |
| Book to average (queue order)       |      0.85 |             106.5 |            7.2 |            13.5 |                 3   |          80.1 |                66.7 |              27.5 |                 5.25 |
| Book to average (queue order)       |      0.9  |             113   |           13.1 |            25.1 |                 5.7 |          84.5 |                70.1 |              27.3 |                 3.75 |
| Book to average (queue order)       |      0.95 |             119.5 |           22.9 |            39.5 |                10.7 |          88.7 |                73.4 |              27.4 |                 2.7  |
| Book to average (queue order)       |      1    |             125.4 |           35.3 |            57.1 |                18.2 |          92.1 |                76.1 |              27.4 |                 1.9  |
| Book to average (ratio order)       |      0.75 |              98   |            1.6 |             4.3 |                 0.7 |          70.6 |                58.5 |              25   |                10.25 |
| Book to average (ratio order)       |      0.8  |             105.3 |            3.6 |             9.2 |                 1.5 |          75.4 |                62.1 |              24.6 |                 7.35 |
| Book to average (ratio order)       |      0.85 |             113.6 |            6.7 |            13.4 |                 2.7 |          80   |                65.4 |              23.3 |                 5.25 |
| Book to average (ratio order)       |      0.9  |             122.2 |           12.4 |            23.7 |                 5.4 |          84.6 |                68.8 |              22.2 |                 3.75 |
| Book to average (ratio order)       |      0.95 |             130.6 |           21.8 |            40.5 |                 9.9 |          88.8 |                71.7 |              21.1 |                 2.7  |
| Book to average (ratio order)       |      1    |             138.3 |           33.8 |            55.7 |                16.3 |          92.2 |                74.1 |              20.2 |                 1.9  |
| Chance-constrained (true durations) |      0.02 |             100.9 |            1.3 |             2.1 |                 0.4 |          71.9 |                59.3 |              24.4 |                10.45 |
| Chance-constrained (true durations) |      0.05 |             109.1 |            3.2 |             4.8 |                 1.1 |          77.2 |                63.3 |              23.5 |                 7.55 |
| Chance-constrained (true durations) |      0.1  |             116.6 |            6.7 |             9.6 |                 2.3 |          81.8 |                66.8 |              22.6 |                 5.3  |
| Chance-constrained (true durations) |      0.2  |             125   |           13.9 |            19.4 |                 5.1 |          86.7 |                70.4 |              21.8 |                 3.5  |
| Chance-constrained (true durations) |      0.3  |             131.8 |           22.3 |            29.3 |                 8.8 |          90.1 |                72.8 |              21   |                 2.45 |
| Chance-constrained (plug-in model)  |      0.02 |             100.6 |            2.2 |             4.7 |                 0.7 |          71.1 |                58.6 |              23.7 |                11.3  |
| Chance-constrained (plug-in model)  |      0.05 |             107.8 |            4.7 |             9.8 |                 1.6 |          76.4 |                62.7 |              23.9 |                 8.4  |
| Chance-constrained (plug-in model)  |      0.1  |             115.4 |            8.4 |            16.3 |                 3.1 |          81   |                66.2 |              22.9 |                 6.1  |
| Chance-constrained (plug-in model)  |      0.2  |             124.3 |           16.4 |            30.8 |                 6.6 |          86.1 |                70   |              21.9 |                 3.75 |
| Chance-constrained (plug-in model)  |      0.3  |             130.4 |           23.9 |            41.9 |                10.4 |          89.2 |                72.2 |              21.5 |                 2.9  |

Extra patients vs book-to-average at the same realised overrun rate

| method    | vs        |   setting |   worlds |   extra_patients |   se |
|:----------|:----------|----------:|---------:|-----------------:|-----:|
| cc-plugin | avg-queue |      0.02 |       13 |             6.53 | 1.77 |
| cc-plugin | avg-queue |      0.05 |       20 |             6.27 | 1.18 |
| cc-plugin | avg-queue |      0.1  |       20 |             7.97 | 1.16 |
| cc-plugin | avg-queue |      0.2  |       20 |             9.19 | 0.98 |
| cc-plugin | avg-queue |      0.3  |       20 |            10.67 | 1.15 |
| cc-oracle | avg-queue |      0.02 |        7 |            10.4  | 1.25 |
| cc-oracle | avg-queue |      0.05 |       20 |             9.97 | 1.04 |
| cc-oracle | avg-queue |      0.1  |       20 |            10.79 | 1.14 |
| cc-oracle | avg-queue |      0.2  |       20 |            11.57 | 0.97 |
| cc-oracle | avg-queue |      0.3  |       20 |            12.61 | 1.01 |
| cc-plugin | avg-ratio |      0.02 |       14 |            -0.39 | 0.95 |
| cc-plugin | avg-ratio |      0.05 |       20 |            -0.5  | 0.53 |
| cc-plugin | avg-ratio |      0.1  |       20 |            -0.62 | 0.3  |
| cc-plugin | avg-ratio |      0.2  |       20 |            -1.36 | 0.38 |
| cc-plugin | avg-ratio |      0.3  |       20 |            -1.5  | 0.2  |
| cc-oracle | avg-ratio |      0.02 |        8 |             4.67 | 0.68 |
| cc-oracle | avg-ratio |      0.05 |       20 |             4.06 | 0.65 |
| cc-oracle | avg-ratio |      0.1  |       20 |             2.72 | 0.56 |
| cc-oracle | avg-ratio |      0.2  |       20 |             1.22 | 0.48 |
| cc-oracle | avg-ratio |      0.3  |       20 |             0.7  | 0.53 |
