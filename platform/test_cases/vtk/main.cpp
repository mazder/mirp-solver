#include <vtkSmartPointer.h>
#include <vtkSphereSource.h>
#include <vtkXMLPolyDataWriter.h>

int main() {
    // Create a sphere source
    vtkSmartPointer<vtkSphereSource> sphereSource = vtkSmartPointer<vtkSphereSource>::New();
    sphereSource->SetRadius(5.0);           // Set the radius of the sphere
    sphereSource->SetThetaResolution(30);   // Set the number of vertical slices
    sphereSource->SetPhiResolution(30);     // Set the number of horizontal slices

    // Write the output to a .vtk file (in XML format)
    vtkSmartPointer<vtkXMLPolyDataWriter> writer = vtkSmartPointer<vtkXMLPolyDataWriter>::New();
    writer->SetFileName("sphere.vtp"); // File extension for VTK XML format
    writer->SetInputConnection(sphereSource->GetOutputPort());

    // Write the file
    writer->Write();

    return 0;
}
