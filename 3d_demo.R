#3D Scatterplot
library(rgl)

args <-commandArgs(trailingOnly=TRUE)

open3d()
given_data<-read.table(args[1], header=FALSE, col.names=c("x","y","z"),sep=",")
#add the camera center (manually inputted)
plot3d(given_data[[1]], given_data[[2]], given_data[[3]], col="red", size=3)
plot3d(3.970854, 3.767054, 0, col="black",size=5,add=TRUE)
plot3d(4.043398, 3.691961, 0, col="black",size=5,add=TRUE)

open3d()
pa2_data<-read.table(args[2], header=FALSE, col.names=c("x","y","z"),sep=",")
plot3d(pa2_data[[1]], pa2_data[[2]], pa2_data[[3]], col="blue", size=3)
invisible(readLines("stdin", n=1))
