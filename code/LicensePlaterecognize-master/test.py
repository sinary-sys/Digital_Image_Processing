def solute(m):
    num = []
    for i in range(2, int(m+1)):
        if m % i == 0:
            num.append(i)
            m = m / i
            break
    print('1')
    print(len(num))


if __name__ == '__main__':
    while True:
        try:
            list1 = list(map(solute, range(10)))
            print(list1)
        except:
            break
