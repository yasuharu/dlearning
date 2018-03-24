
#include <stdint.h>

#include <string>
#include <vector>
#include <memory>

struct Image
{
	public:
		Image(int32_t image_size)
			: image_size(image_size)
		{
			image = new uint8_t[image_size];
		}

		virtual ~Image()
		{
			delete[] image;
		}

		uint8_t *image;
		const uint32_t image_size;
};

class MnistLoader
{
	public:
		bool LoadImage(std::string file_name, std::vector<std::shared_ptr<Image> > &image_list);
		bool LoadLabel(std::string file_name, std::vector<uint8_t> &label_list);
};

