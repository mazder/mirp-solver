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
        void convert_dicom_to_png(const std::string& basePath);
        void create_png_to_folds(const std::string& basePath);
        void create_png_to_folds_rr(const std::string& basePath);
        void ogo_callibration(const std::string& basePath);
    protected:

    private:
};

#endif // PREPROCESSING_HPP
