# def hash_list(data_list):
#     joined_data = ''.join(map(str, data_list))
#     return hashlib.sha256(joined_data.encode()).hexdigest()
#
#
# hash_before = 'ceb34d1f825005a0871df6c759b5d9f2dc5dcc866f8e9b57b5a9383ea0752e66'
# hash_after = hash_list(all_other_parameters_up)
# # print(hash_after)
# if hash_before == hash_after:
#     print('\033[32mВсе ок')
# # Сравниваем хеш-суммы
# assert hash_before == hash_after, "Данные изменились!"
#
# tickers = ['BTCUSDT']
# timeframes = ['30MINUTE']
#
# s_date = "2022-01-01"
# u_date = "2023-07-05"
