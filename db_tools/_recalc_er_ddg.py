import math

ee = 99.5

er = (50+(ee/2))/(50-(ee/2))

ddG_er = ((25+273.15)*(8.314))*math.log(er)*(0.000239)

print(f'er is {er}, ddG_er is {ddG_er}')