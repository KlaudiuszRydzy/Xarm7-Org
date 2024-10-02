import xarm_gym as xg
from xarm_gym import utils as xgu

# TEST CartesianBoundary
b = xgu.CartesianBoundary(
    xgu.PartialRobotState(cartesian=[1, 2, 3]),
    xgu.PartialRobotState(cartesian=[4, 5, 6]),
)

state = xgu.PartialRobotState(cartesian=[4, 5, 6])

print(state.cartesian)

print(b.contains(state))

nb = xgu.NOTBoundary(b)

print(nb)
print(nb.contains(state))

b = xgu.ORBoundary([b, xgu.NOTBoundary(b)])

print(b)
print(b.contains(state))

other = xgu.CartesianBoundary(
    xgu.PartialRobotState(cartesian=[7, 8, 9]),
    xgu.PartialRobotState(cartesian=[10, 11, 12]),
)


print(xgu.NOTBoundary(other))
print(xgu.NOTBoundary(other).contains(state))


# TEST JointBoundary
b = xgu.JointBoundary(
    xgu.PartialRobotState(joints=[1, 2, 3, 4, 5, 6, 7]),
    xgu.PartialRobotState(joints=[8, 9, 10, 11, 12, 13, 14]),
)
state = xgu.PartialRobotState(joints=[8, 9, 10, 11, 12, 13, 14])

print(state.joints)
print(b.contains(state))
