function a = getAccuracy(p, y)
  a =  mean(double(p == y)) * 100;
endfunction
