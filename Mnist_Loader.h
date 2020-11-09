#include <fstream>

#ifndef MNIST_LOADER_H
#define MNIST_LOADER_H

#include <vector>
#include <string>

class mnist_loader {
private:
	std::vector<std::vector<double>> m_images;
	std::vector<int> m_labels;
	int m_size;
	int m_rows;
	int m_cols;

	int ReverseInt(int i)
	{
		unsigned char ch1, ch2, ch3, ch4;
		ch1 = i & 255;
		ch2 = (i >> 8) & 255;
		ch3 = (i >> 16) & 255;
		ch4 = (i >> 24) & 255;
		return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
	}
	void load_images(std::string file, int num = 0);
	void load_labels(std::string file, int num = 0);
	int  to_int(char* p);

public:
	mnist_loader(std::string image_file, std::string label_file, int num);
	mnist_loader(std::string image_file, std::string label_file);
	~mnist_loader();

	int size() { return m_size; }
	int rows() { return m_rows; }
	int cols() { return m_cols; }

	std::vector<double> images(int id) { return m_images[id]; }
	int labels(int id) { return m_labels[id]; }
};


#endif
