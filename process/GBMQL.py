import QuantLib as ql



if __name__ == "__main__":
   init_val = 100
   mu = 0.01
   sigma = 0.2
   process = ql.GeometricBrownianMotionProcess(init_val, mu, sigma)



   print(ql.proce)