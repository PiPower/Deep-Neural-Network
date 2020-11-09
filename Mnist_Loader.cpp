#include "Mnist_Loader.h"
#include <assert.h>

using namespace std;
mnist_loader::mnist_loader(std::string image_file,
    std::string label_file,
    int num) :
    m_size(0),
    m_rows(0),
    m_cols(0)
{
    load_images(image_file, num);
    load_labels(label_file, num);
}

mnist_loader::mnist_loader(std::string image_file,
    std::string label_file) :
    mnist_loader(image_file, label_file, 0)
{
    // empty
}

mnist_loader::~mnist_loader()
{
    // empty
}

int
mnist_loader::to_int(char* p)
{
    return ((p[0] & 0xff) << 24) | ((p[1] & 0xff) << 16) |
        ((p[2] & 0xff) << 8) | ((p[3] & 0xff) << 0);
}

void
mnist_loader::load_images(std::string image_file, int num)
{
    std::ifstream ifs(image_file.c_str(), std::ios::in | std::ios::binary);

	if (ifs.is_open())
	{
		int magic_number = 0;
		int number_of_images = 0;
		int n_rows = 0;
		int n_cols = 0;
        ifs.read((char*)&magic_number, sizeof(magic_number));
		magic_number = ReverseInt(magic_number);
        ifs.read((char*)&number_of_images, sizeof(number_of_images));
		number_of_images = ReverseInt(number_of_images);
        ifs.read((char*)&n_rows, sizeof(n_rows));
		n_rows = ReverseInt(n_rows);
        ifs.read((char*)&n_cols, sizeof(n_cols));
		n_cols = ReverseInt(n_cols);
		for (int i = 0; i < num; i++)
		{
            vector<double> img;
			for (int r = 0; r < n_rows; r++)
			{
				for (int c = 0; c < n_cols; c++)
				{
					unsigned char temp = 0;
                    ifs.read((char*)&temp, sizeof(temp));
					int l = (n_rows * r) + c;
                    img.push_back((double)temp / 255.0);
				}
			}
            m_images.push_back(img);
		}
	}
}

void
mnist_loader::load_labels(std::string label_file, int num)
{
    std::ifstream ifs(label_file.c_str(), std::ios::in | std::ios::binary);
    char p[4];

    ifs.read(p, 4);
    int magic_number = to_int(p);
    assert(magic_number == 0x801);

    ifs.read(p, 4);
    int size = to_int(p);
    // limit
    if (num != 0 && num < m_size) size = num;

    for (int i = 0; i < size; ++i) {
        ifs.read(p, 1);
        int label = p[0];
        m_labels.push_back(label);
    }

    ifs.close();
}
