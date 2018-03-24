#include "mnist_loader.h"

namespace
{
	bool read_uint32_t(FILE *fp, uint32_t *retval)
	{
		int val;
		int count = 0;
		while((val = fgetc(fp)) != EOF)
		{
			(*retval) = ((*retval) << 8) | (uint8_t)val;

			count++;
			if(count == 4)	break;
		}

		if(val == EOF)
		{
			return false;
		}else
		{
			return true;
		}
	}

	bool read_uint8_t(FILE *fp, uint8_t *retval)
	{
		int val;
		if((val = fgetc(fp)) != EOF)
		{
			(*retval) = (uint8_t)val;
			return true;
		}else
		{
			return false;
		}
	}
};

bool MnistLoader::LoadImage(std::string file_name, std::vector<std::shared_ptr<Image> > &image_list)
{
	FILE *fp = fopen(file_name.c_str(), "rb");
	if(fp == NULL)
	{
		perror("can't open file.");
		return false;
	}

	uint32_t magic;
	uint32_t image_num;
	uint32_t image_height;
	uint32_t image_width;

/* Data format
[offset] [type]          [value]          [description] 
0000     32 bit integer  0x00000803(2051) magic number 
0004     32 bit integer  60000            number of images 
0008     32 bit integer  28               number of rows 
0012     32 bit integer  28               number of columns 
0016     unsigned byte   ??               pixel 
0017     unsigned byte   ??               pixel 
*/
	if(read_uint32_t(fp, &magic) == false)
	{
		printf("[ERROR] can't read magic num. magic = 0x%08x\n", magic);
		return false;
	}

	if(magic != 0x00000803)
	{
		printf("[ERROR] wrong magic num.\n");
		return false;
	}

	if(read_uint32_t(fp, &image_num) == false)
	{
		printf("[ERROR] can't read image num.\n");
		return false;
	}

	if(read_uint32_t(fp, &image_height) == false)
	{
		printf("[ERROR] can't read image height.\n");
		return false;
	}

	if(read_uint32_t(fp, &image_width) == false)
	{
		printf("[ERROR] can't read image width.\n");
		return false;
	}

	const uint32_t pixel_num = image_height * image_width;
	for(int image_index = 0 ; image_index < image_num ; image_index++)
	{
		std::shared_ptr<Image> image = std::make_shared<Image>(pixel_num);

//		printf("[INFO] read %d image.\n", image_index);

		for(int pixel_index = 0 ; pixel_index < pixel_num ; pixel_index++)
		{
//			printf("[INFO] read %d pixel.\n", pixel_index);

			uint8_t val;
			if(read_uint8_t(fp, &val) == false)
			{
				printf("[ERROR] can't read pixels.\n");
				return false;
			}
			image->image[pixel_index] = val;
		}

		image_list.push_back(image);
	}

	// checking whole data was read
	if(fgetc(fp) != EOF)
	{
		return false;
	}

	return true;
}

bool MnistLoader::LoadLabel(std::string file_name, std::vector<uint8_t> &label_list)
{
	FILE *fp = fopen(file_name.c_str(), "rb");
	if(fp == NULL)
	{
		perror("can't open file.");
		return false;
	}

	uint32_t magic;
	uint32_t image_num;

/* Data format
[offset] [type]          [value]          [description] 
0000     32 bit integer  0x00000801(2049) magic number (MSB first) 
0004     32 bit integer  60000            number of items 
0008     unsigned byte   ??               label 
0009     unsigned byte   ??               label 
*/
	if(read_uint32_t(fp, &magic) == false)
	{
		printf("[ERROR] can't read magic num. magic = 0x%08x\n", magic);
		return false;
	}

	if(magic != 0x00000801)
	{
		printf("[ERROR] wrong magic num.\n");
		return false;
	}

	if(read_uint32_t(fp, &image_num) == false)
	{
		printf("[ERROR] can't read image num.\n");
		return false;
	}

	for(int image_index = 0 ; image_index < image_num ; image_index++)
	{
		uint8_t val;
		if(read_uint8_t(fp, &val) == false)
		{
			printf("[ERROR] can't read pixels.\n");
			return false;
		}
		label_list.push_back(val);
	}

	// checking whole data was read
	if(fgetc(fp) != EOF)
	{
		return false;
	}

	return true;
}

