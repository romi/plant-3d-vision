FROM geki/colmap
# geki/colmap is based on nvidia/cuda:10.1-devel-ubuntu18.04

RUN apt-get update \
	&& apt-get -y install python3.7

RUN apt-get update \
	&& apt-get -y install python3.7-dev

RUN apt-get update \
	&& apt-get install wget

RUN apt-get update \
	&& apt-get -y install python3-pip

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.7 10

RUN apt-get update && apt-get install -y --no-install-recommends \
        ocl-icd-libopencl1 opencl-headers\
        clinfo && \
		apt-get clean && \
		rm -rf /var/lib/apt/lists/*

#Todo : use ldconfig
RUN ln -s /usr/lib/x86_64-linux-gnu/libOpenCL.so.1 /usr/lib/libOpenCL.so

RUN git clone https://github.com/inducer/pyopencl && \
	cd pyopencl/ && \
	git submodule update --init && \
	python3.7 -m pip install pybind11 mako && \
	./configure.py --cl-pretend-version=1.2 && \
	python3.7 setup.py install

RUN mkdir -p /etc/OpenCL/vendors && \
    echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

RUN cd .. && \
	git clone https://github.com/romi/data-storage.git && \
	cd data-storage/ && \
	git checkout dev && \
	python3.7 setup.py develop

# TODO : voir si c'est essentiel ou pas?
RUN cd .. && \
	git clone https://github.com/romi/lettucethink-python && \
	cd lettucethink-python/ && \
	git checkout dev && \
	python3.7 setup.py develop

RUN cd .. && \
	git clone https://github.com/romi/cgal_bindings_skeletonization && \
	cd cgal_bindings_skeletonization/ && \
	python3.7 setup.py develop

RUN python3.7 -m pip install pyyaml
RUN python3.7 -m pip install pyqt5==5.14

RUN cd .. && \
	git clone https://github.com/romi/Segmentation && \
	cd Segmentation/ && \
	git checkout dev && \
	python3.7 setup.py develop

RUN cd .. && \
	git clone https://github.com/romi/romiscanner.git && \
	cd romiscanner/ && \
	python3.7 setup.py develop

RUN cd .. && \
	git clone https://github.com/romi/Scan3D && \
	cd Scan3D/ && \
	git checkout dev && \
	python3.7 -m pip install -r requirements.txt && \
	python3.7 setup.py develop

ENV DB_LOCATION="/home/${SETUSER}/db_test"

CMD ["/bin/bash", "-c", "source ~/.bashrc && cd Scan3D/tests/ && ./check_pipe.sh"]