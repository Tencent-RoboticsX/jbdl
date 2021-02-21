function Xinv = InverseMotionSpace(X)
E = X(1:3,1:3);
r = X(4:6,1:3);
Xinv = [E',zeros(3);r',E'];
end