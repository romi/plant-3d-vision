FROM geki/colmap
# geki/colmap is based on nvidia/cuda:10.1-devel-ubuntu18.04

RUN apt-get update && apt-get install -y \
	python3.7 python3.7-dev python3-pip wget && \
	apt-get clean && \
	rm -rf /var/lib/apt/lists/* && \
	update-alternatives --install /usr/bin/python python /usr/bin/python3.7 10

RUN apt-get update && apt-get install -y --no-install-recommends \
        ocl-icd-libopencl1 opencl-headers\
        clinfo && \
		apt-get clean && \
		rm -rf /var/lib/apt/lists/* && \
		ln -s /usr/lib/x86_64-linux-gnu/libOpenCL.so.1 /usr/lib/libOpenCL.so

RUN git clone https://github.com/inducer/pyopencl && \
	cd pyopencl/ && \
	git submodule update --init && \
	python -m pip install pybind11 mako && \
	./configure.py --cl-pretend-version=1.2 && \
	python setup.py install

RUN mkdir -p /etc/OpenCL/vendors && \
    echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

RUN cd .. && \
	git clone https://github.com/romi/data-storage.git && \
	cd data-storage/ && \
	git checkout dev && \
	python setup.py develop

RUN cd .. && \
	git clone https://github.com/romi/cgal_bindings_skeletonization && \
	cd cgal_bindings_skeletonization/ && \
	python setup.py develop

RUN cd .. && \
	git clone https://github.com/romi/Segmentation && \
	cd Segmentation/ && \
	git checkout dev && \
	python setup.py develop

RUN cd .. && \
	git clone https://github.com/romi/romiscanner.git && \
	cd romiscanner/ && \
	python setup.py develop

RUN cd .. && \
	git clone https://github.com/romi/Scan3D && \
	cd Scan3D/ && \
	git checkout dev && \
	python -m pip install -r requirements.txt && \
	python setup.py develop

ENV DB_LOCATION="/home/${SETUSER}/db_test"

CMD ["/bin/bash", "-c", "cd .."]
