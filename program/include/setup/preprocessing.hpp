#ifndef PREPROCESSING_HPP
#define PREPROCESSING_HPP

#include <vector>
#include <string>
#include <iostream>

class preprocessing
{
    public:
        preprocessing();
        virtual ~preprocessing();
        void dicom_to_png(const std::string& in_path,const std::string& out_path);
    protected:

    private:
};

#endif // PREPROCESSING_HPP
