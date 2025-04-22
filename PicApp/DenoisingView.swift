import SwiftUI
import CoreML
import PhotosUI
import CoreImage
import CoreImage.CIFilterBuiltins

class PickerDelegate: NSObject, PHPickerViewControllerDelegate {
    let completion: (UIImage?) -> Void

    init(completion: @escaping (UIImage?) -> Void) {
        self.completion = completion
    }

    func picker(_ picker: PHPickerViewController, didFinishPicking results: [PHPickerResult]) {
        picker.dismiss(animated: true)
        guard let result = results.first else {
            completion(nil)
            return
        }

        if result.itemProvider.canLoadObject(ofClass: UIImage.self) {
            result.itemProvider.loadObject(ofClass: UIImage.self) { [weak self] object, _ in
                DispatchQueue.main.async {
                    self?.completion(object as? UIImage)
                }
            }
        } else {
            completion(nil)
        }
    }
}


struct DenoisingView: View {
    @State private var selectedModel: String? = nil
    @State private var selectedClassicMethod: String = "None"
    @State private var blurRadius: Float = 5.0
    @State private var isProcessing = false
    @State private var resultImage: UIImage? = nil
    @State private var inferenceTime: TimeInterval? = nil
    @State private var inputImage: UIImage? = nil
    @State private var pickerDelegate: PickerDelegate? = nil

    let models = ["DnCNN", "EDD"]
    let classicMethods = ["None", "Gaussian Blur", "Median Filter", "Bilateral Filter"]

    var body: some View {
        ScrollView {
            VStack {
                // ML Model Selection
                Text("Neural Network Model")
                    .font(.headline)
                    .padding(.top)

                Picker("Select Model", selection: Binding(
                    get: { selectedModel ?? "" },
                    set: { selectedModel = $0.isEmpty ? nil : $0 }
                )) {
                    Text("Select a model").tag("")
                    ForEach(models, id: \.self) { model in
                        Text(model).tag(model)
                    }
                }
                .pickerStyle(SegmentedPickerStyle())
                .padding(.horizontal)

                // Classical Methods Selection
                Text("Classical Method")
                    .font(.headline)
                    .padding(.top)

                Picker("Select Method", selection: $selectedClassicMethod) {
                    ForEach(classicMethods, id: \.self) { method in
                        Text(method).tag(method)
                    }
                }
                .pickerStyle(SegmentedPickerStyle())
                .padding(.horizontal)

                // Blur Radius Slider (only shown for relevant methods)
                if selectedClassicMethod == "Gaussian Blur" || selectedClassicMethod == "Bilateral Filter" {
                    VStack {
                        Text("Filter Radius: \(Int(blurRadius))")
                            .font(.subheadline)

                        Slider(value: $blurRadius, in: 1...25, step: 1)
                            .padding(.horizontal)
                    }
                    .padding(.top, 5)
                }

                Button(action: loadImage) {
                    Text("Load Image")
                        .padding()
                        .background(Color.blue)
                        .foregroundColor(.white)
                        .cornerRadius(10)
                }
                .padding()

                if isProcessing {
                    ProgressView("Processing...")
                        .padding()
                }

                if let resultImage = resultImage {
                    Text("Result Image")
                        .font(.subheadline)
                        .foregroundColor(.gray)
                        .padding(.top)
                    Image(uiImage: resultImage)
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                        .frame(height: 300)
                        .padding()

                    if let time = inferenceTime {
                        Text("Processing Time: \(String(format: "%.2f", time)) seconds")
                            .font(.subheadline)
                            .foregroundColor(.gray)
                            .padding()
                    }
                } else if let inputImage = inputImage {
                    // Show original image if we have one but no result yet
                    Text("Original Image")
                        .font(.subheadline)
                        .foregroundColor(.gray)

                    Image(uiImage: inputImage)
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                        .frame(height: 300)
                        .padding()
                } else {
                    Text("Processed image will appear here.")
                        .foregroundColor(.gray)
                }

                if inputImage != nil {
                    Button(action: runDenoising) {
                        Text("Run Denoising")
                            .padding()
                            .background(Color.green)
                            .foregroundColor(.white)
                            .cornerRadius(10)
                    }
                    .padding(.bottom)
                }
            }
        }
        .navigationTitle("Image Denoising")
    }


    func loadImage() {
        let configuration = PHPickerConfiguration(photoLibrary: .shared())
        let picker = PHPickerViewController(configuration: configuration)

        let delegate = PickerDelegate { image in
            DispatchQueue.main.async {
                self.inputImage = image
                self.resultImage = nil // Clear previous result
            }
        }

        self.pickerDelegate = delegate
        picker.delegate = delegate

        if let windowScene = UIApplication.shared.connectedScenes.first as? UIWindowScene,
           let rootViewController = windowScene.windows.first?.rootViewController {
            rootViewController.present(picker, animated: true)
        }
    }

    func runDenoising() {
            guard let inputImage = inputImage else {
                print("No input image loaded.")
                return
            }
            isProcessing = true
            resultImage = nil
            inferenceTime = nil

            DispatchQueue.global(qos: .userInitiated).async {
                let startTime = CFAbsoluteTimeGetCurrent()

                var processedImage: UIImage?

                if self.selectedClassicMethod != "None" {
                    processedImage = self.applyClassicDenoising(to: inputImage)
                } else if let selectedModel = self.selectedModel {
                    processedImage = self.applyMLDenoising(to: inputImage, model: selectedModel)
                } else {
                    print("No denoising method selected.")
                }

                let endTime = CFAbsoluteTimeGetCurrent()

                DispatchQueue.main.async {
                    self.resultImage = processedImage
                    self.inferenceTime = endTime - startTime
                    self.isProcessing = false
                    print("Processing finished. Time: \(String(format: "%.2f", self.inferenceTime ?? 0)) seconds.")
                    if self.resultImage == nil {
                        print("Result image is nil. Check for errors in processing.")
                    } else {
                        print("Result image dimensions: \(self.resultImage?.size ?? .zero)")
                    }
                }
            }
        }

    func applyClassicDenoising(to image: UIImage) -> UIImage? {
   guard let ciImage = CIImage(image: image) else { return nil }
        let context = CIContext()

                var outputCIImage: CIImage?
            
   switch selectedClassicMethod {
   case "Gaussian Blur":
   let filter = CIFilter.gaussianBlur()
   filter.inputImage = ciImage
   filter.radius = Float(blurRadius)
   outputCIImage = filter.outputImage
             case "Median Filter":
   // Note: CIMedianFilter isn't available in Core Image
    // Using a workaround with CIFilter.name approach
   if let filter = CIFilter(name: "CIMedianFilter") {
      filter.setValue(ciImage, forKey: kCIInputImageKey)
   outputCIImage = filter.outputImage
   } else {
   // Fallback to noise reduction if median filter isn't available
    let filter = CIFilter.noiseReduction()
   filter.inputImage = ciImage
        filter.noiseLevel = 0.02
   filter.sharpness = 0.4
    outputCIImage = filter.outputImage
    }
    case "Bilateral Filter":
   // Core Image doesn't have a direct bilateral filter
    // Using a combination of noise reduction and edge preservation
   let noiseFilter = CIFilter.noiseReduction()
   noiseFilter.inputImage = ciImage
   noiseFilter.noiseLevel = Float(blurRadius) / 50.0
   noiseFilter.sharpness = 0.5

   if let noiseFiltered = noiseFilter.outputImage {
   // Add a bit of edge preservation
   let sharpenFilter = CIFilter.sharpenLuminance()
   sharpenFilter.inputImage = noiseFiltered
   sharpenFilter.sharpness = Float(blurRadius) / 10.0
   outputCIImage = sharpenFilter.outputImage
   } else {
   outputCIImage = noiseFilter.outputImage
   }

   default:
   outputCIImage = ciImage
                    
                }
            
        guard let outputCIImage = outputCIImage,
            let cgImage = context.createCGImage(outputCIImage, from: outputCIImage.extent) else {
            return nil
            }

    return UIImage(cgImage: cgImage)}
    // Apply ML-based denoising
    func applyMLDenoising(to image: UIImage, model: String) -> UIImage? {
        let modelName: String
        switch model {
        case "DnCNN":
            modelName = "dncnn-epoch=028-val_loss=0.0001"
        case "EDD":
            modelName = "DnCNN_SIDD_small_50x50_experiment_1_traced"
        default:
            print("Invalid model selected: \(model)")
            return nil
        }

        guard let modelURL = Bundle.main.url(forResource: modelName, withExtension: "mlmodelc") else {
            print("Failed to find model file: \(modelName).mlmodelc in bundle.")
            return nil
        }

        guard let mlModel = try? MLModel(contentsOf: modelURL) else {
            print("Failed to load or compile model from URL: \(modelURL).")
            return nil
        }

        guard let patches = preprocessImage(image) else {
            print("Failed to preprocess image into patches.")
            return nil
        }

        print("Number of patches created: \(patches.count)")

        var processedPatches: [UIImage] = []
        for (index, patch) in patches.enumerated() {
            print("Processing patch \(index + 1) of \(patches.count)")
            guard let inputPixelBuffer = patch.toCVPixelBuffer() else {
                print("Failed to create pixel buffer for input patch \(index).")
                continue
            }

            do {
                let width = 50
                let height = 50
                let inputArray = try MLMultiArray(shape: [1, 3, NSNumber(value: height), NSNumber(value: width)], dataType: .float32)

                CVPixelBufferLockBaseAddress(inputPixelBuffer, .readOnly)
                if let baseAddress = CVPixelBufferGetBaseAddress(inputPixelBuffer) {
                    let buffer = baseAddress.assumingMemoryBound(to: UInt8.self)

                    for y in 0..<height {
                        for x in 0..<width {
                            let offset = y * CVPixelBufferGetBytesPerRow(inputPixelBuffer) + x * 4

                            let r = Float(buffer[offset + 1]) / 255.0
                            let g = Float(buffer[offset + 2]) / 255.0
                            let b = Float(buffer[offset + 3]) / 255.0

                            inputArray[[0, 0, NSNumber(value: y), NSNumber(value: x)]] = NSNumber(value: r)
                            inputArray[[0, 1, NSNumber(value: y), NSNumber(value: x)]] = NSNumber(value: g)
                            inputArray[[0, 2, NSNumber(value: y), NSNumber(value: x)]] = NSNumber(value: b)
                            if index == 0 && x < 5 && y < 5 {
                                print("Input Patch (0,0) R:\(r) G:\(g) B:\(b)")
                            }
                        }
                    }
                }
                CVPixelBufferUnlockBaseAddress(inputPixelBuffer, .readOnly)

                let input = try MLDictionaryFeatureProvider(dictionary: ["input": inputArray])
                print("Input MLMultiArray shape: \(inputArray.shape)")
                
                let prediction = try mlModel.prediction(from: input)
                print("Prediction features: \(prediction.featureNames)")

                if let outputArray = prediction.featureValue(for: "var_252")?.multiArrayValue {
                    print("Output MLMultiArray received for patch \(index), shape: \(outputArray.shape).")
                    if index == 0 {
                        var firstFewValues = ""
                        for i in 0..<min(5, outputArray.count) {
                            firstFewValues += "\(outputArray[i].floatValue), "
                        }
                        var minValue: Float = Float.greatestFiniteMagnitude
                            var maxValue: Float = Float.leastNormalMagnitude

                            let count = inputArray.count
                            let pointer = inputArray.dataPointer.assumingMemoryBound(to: Float.self) // Presupunem ca modelul returneaza Float. Va trebui ajustat daca e alt tip.

                            for i in 0..<count {
                                let value = pointer[i]
                                minValue = min(minValue, value)
                                maxValue = max(maxValue, value)
                            }
                            print("Output Patch (0,0) - Min Value: \(minValue), Max Value: \(maxValue)")
                        print("Output Patch (0,0) - First 5 values: \(firstFewValues)")
                    }

                    let processedMLMultiArray = try MLMultiArray(shape: [1, 3, NSNumber(value: height), NSNumber(value: width)], dataType: .float32)
                    for i in 0..<outputArray.count {
                        processedMLMultiArray[i] = NSNumber(value: inputArray[i].floatValue)
//                        processedMLMultiArray[i] = NSNumber(value: scaled)

                        if index == 0 && i < 5 {
                            print("processedMLMultiArray (raw)[\(index)] = \(processedMLMultiArray[index])")
                        }
                    }

                    if let outputPixelBuffer = processedMLMultiArray.toCVPixelBuffer(width: width, height: height) {
                        if let processedImage = UIImage(pixelBuffer: outputPixelBuffer) {
                    
                            processedPatches.append(processedImage)
                        } else {
                            print("Failed to create UIImage from output pixel buffer for patch \(index).")
                        }
                    } else {
                        print("Failed to convert output MLMultiArray to CVPixelBuffer for patch \(index).")
                    }
                } else {
                    print("No 'var_252' output received from model for patch \(index).")
                }

            } catch {
                print("Error processing patch \(index): \(error)")
            }
        }

        print("Number of processed patches: \(processedPatches.count)")

        let reconstructedImage = reconstructImage(from: processedPatches, originalSize: image.size)
        if reconstructedImage == nil {
            print("Failed to reconstruct image from processed patches.")
        } else {
            print("Reconstructed image dimensions: \(reconstructedImage?.size ?? .zero)")
        }
        return reconstructedImage
    }

    func preprocessImage(_ image: UIImage) -> [UIImage]? {
        guard let cgImage = image.cgImage else {
            print("Failed to get CGImage from input UIImage for preprocessing.")
            return nil
        }
        let width = cgImage.width
        let height = cgImage.height
        let patchSize = 50
        var patches: [UIImage] = []
        let colorSpace = CGColorSpaceCreateDeviceRGB()

        print("Preprocessing image of size: \(width)x\(height) with patch size: \(patchSize)")

        for y in stride(from: 0, to: height, by: patchSize) {
            for x in stride(from: 0, to: width, by: patchSize) {
                let rect = CGRect(x: x, y: y, width: patchSize, height: patchSize)
                if let croppedCGImage = cgImage.cropping(to: rect) {
                    print("Created patch at x:\(x), y:\(y) with size: \(croppedCGImage.width)x\(croppedCGImage.height)")
                    patches.append(UIImage(cgImage: croppedCGImage))
                } else {
                    let remainingWidth = min(patchSize, width - x)
                    let remainingHeight = min(patchSize, height - y)
                    if remainingWidth > 0 && remainingHeight > 0 {
                        let clippedRect = CGRect(x: x, y: y, width: remainingWidth, height: remainingHeight)
                        if let clippedCGImage = cgImage.cropping(to: clippedRect) {
                            print("Created clipped patch at x:\(x), y:\(y) with size: \(clippedCGImage.width)x\(clippedCGImage.height)")
                            let context = CGContext(
                                data: nil,
                                width: patchSize,
                                height: patchSize,
                                bitsPerComponent: 8,
                                bytesPerRow: patchSize * 4,
                                space: colorSpace,
                                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
                            )
                            context?.setFillColor(UIColor.black.cgColor)
                            context?.fill(CGRect(x: 0, y: 0, width: patchSize, height: patchSize))
                            context?.draw(clippedCGImage, in: CGRect(origin: .zero, size: CGSize(width: remainingWidth, height: remainingHeight)))
                            if let patchedCGImage = context?.makeImage() {
                                patches.append(UIImage(cgImage: patchedCGImage))
                            }
                        }
                    }
                }
            }
        }
        print("Number of patches after cropping: \(patches.count)")
        return patches
    }

    func reconstructImage(from patches: [UIImage], originalSize: CGSize) -> UIImage? {
            let width = Int(originalSize.width)
            let height = Int(originalSize.height)
            let patchSize = 50
            let colorSpace = CGColorSpaceCreateDeviceRGB()

            guard let context = CGContext(data: nil,
                                          width: width,
                                          height: height,
                                          bitsPerComponent: 8,
                                          bytesPerRow: width * 4,
                                          space: colorSpace,
                                          bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else {
                print("Failed to create graphics context for image reconstruction.")
                return nil
            }
            context.setFillColor(UIColor.white.cgColor)
            context.fill(CGRect(origin: .zero, size: originalSize))

            var patchIndex = 0
            for y in stride(from: 0, to: height, by: patchSize) { // Iterăm pe rânduri
                for x in stride(from: 0, to: width, by: patchSize) { // Apoi pe coloane
                    if patchIndex < patches.count, let patchCGImage = patches[patchIndex].cgImage {
                        let drawWidth = min(patchSize, width - x)
                        let drawHeight = min(patchSize, height - y)

                        context.draw(patchCGImage, in: CGRect(x: CGFloat(x), y: CGFloat(y), width: CGFloat(drawWidth), height: CGFloat(drawHeight)))

                        patchIndex += 1
                    } else {
                        print("Warning: Not enough processed patches to reconstruct the full image at index \(patchIndex) (x:\(x), y:\(y)).")
                    }
                }
            }

            if let finalCGImage = context.makeImage() {
                print("Successfully reconstructed the image.")
                return UIImage(cgImage: finalCGImage)
            } else {
                print("Failed to create the final CGImage from the context.")
                return nil
            }
        }
                                      }

                                      extension UIImage {
                                          func toCVPixelBuffer() -> CVPixelBuffer? {
                                              let width = Int(self.size.width)
                                              let height = Int(self.size.height)

                                              var pixelBuffer: CVPixelBuffer?
                                              let status = CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_32ARGB, nil, &pixelBuffer)

                                              guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
                                                  print("Failed to create CVPixelBuffer.")
                                                  return nil
                                              }

                                              CVPixelBufferLockBaseAddress(buffer, [])
                                              let pxdata = CVPixelBufferGetBaseAddress(buffer)

                                              let rgbColorSpace = CGColorSpaceCreateDeviceRGB()
                                              guard let context = CGContext(data: pxdata,
                                                                            width: width,
                                                                            height: height,
                                                                            bitsPerComponent: 8,
                                                                            bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
                                                                            space: rgbColorSpace,
                                                                            bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue) else {
                                                  CVPixelBufferUnlockBaseAddress(buffer, [])
                                                  print("Failed to create CGContext for CVPixelBuffer conversion.")
                                                  return nil
                                              }

                                              UIGraphicsPushContext(context)
                                              self.draw(in: CGRect(x: 0, y: 0, width: width, height: height))
                                              UIGraphicsPopContext()

                                              CVPixelBufferUnlockBaseAddress(buffer, [])

                                              return buffer
                                          }

                                          convenience init?(pixelBuffer: CVPixelBuffer) {
                                              let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
                                              let context = CIContext(options: nil)

                                              guard let cgImage = context.createCGImage(ciImage, from: ciImage.extent) else {
                                                  print("Failed to create CGImage from CVPixelBuffer.")
                                                  return nil
                                              }

                                              self.init(cgImage: cgImage, scale: 1.0, orientation: .up)
                                          }
                                      }

                                      extension MLMultiArray {
                                          func toCVPixelBuffer(width: Int, height: Int) -> CVPixelBuffer? {
                                              print("Inside toCVPixelBuffer. MLMultiArray shape: \(self.shape), dataType: \(self.dataType), count: \(self.count)")

                                              let pixelFormatType = kCVPixelFormatType_32ARGB
                                              var pixelBuffer: CVPixelBuffer?
                                              let status = CVPixelBufferCreate(kCFAllocatorDefault, width, height, pixelFormatType, nil, &pixelBuffer)
                                              guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
                                                  print("Error: Could not create pixel buffer.")
                                                  return nil
                                              }

                                              CVPixelBufferLockBaseAddress(buffer, .init(rawValue: 0))
                                              let baseAddress = CVPixelBufferGetBaseAddress(buffer)
                                              let floatChannelCount = 4
                                              let bytesPerRow = CVPixelBufferGetBytesPerRow(buffer)

                                              if self.dataType == .float32 && self.shape.count == 4 && self.shape[0].intValue == 1 && self.shape[1].intValue == 3 && self.shape[2].intValue == height && self.shape[3].intValue == width {
                                                  print("Assuming MLMultiArray format: [1, 3, H, W] (RGB)")
                                                  let ptr = UnsafeMutablePointer<Float>(OpaquePointer(baseAddress))
                                                  for y in 0..<height {
                                                      for x in 0..<width {
                                                          let r = max(0, min(255, self[[0, 0, NSNumber(value: y), NSNumber(value: x)]].floatValue)) * 255
                                                          let g = max(0, min(255, self[[0, 1, NSNumber(value: y), NSNumber(value: x)]].floatValue)) * 255
                                                          let b = max(0, min(255, self[[0, 2, NSNumber(value: y), NSNumber(value: x)]].floatValue)) * 255

                                                          let byteOffset = y * bytesPerRow + x * floatChannelCount
                                                          var pixel: UInt32 = 0xFF000000 | (UInt32(r) << 16) | (UInt32(g) << 8) | UInt32(b)
                                                          baseAddress?.advanced(by: byteOffset).assumingMemoryBound(to: UInt32.self).pointee = pixel
                                                          if x < 5 && y < 5 {
                                                              print("toCVPixelBuffer (RGB) R:\(r) G:\(g) B:\(b)")
                                                          }
                                                      }
                                                  }
                                              } else {
                                                  print("Error: Unexpected data type or shape in MLMultiArray for CVPixelBuffer conversion.")
                                                  CVPixelBufferUnlockBaseAddress(buffer, [])
                                                  return nil
                                              }

                                              CVPixelBufferUnlockBaseAddress(buffer, [])
                                              return buffer
                                          }
                                      }
