function r = TransformToPosition(X)
E = X(1:3,1:3);
rx = -E'*X(4:6,1:3);
r = [-rx(2,3);rx(1,3);-rx(1,2)];
end