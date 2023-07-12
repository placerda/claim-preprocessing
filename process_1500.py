from util.general_utilities import load_image, get_filename, count_digits
from util.image_analysis import crop_from_qrcode, analyze_document_rest, concatenate_lines
from util.image_processing import extract_roi, remove_blobs, remove_hlines
from util.field_extraction import extract_insured_id, extract_date, is_valid_date, llm_extract_date, extract_cpthcpccode, extract_charges
import cv2
import csv
from datetime import datetime
from glob import glob
import imutils
import logging
import numpy as np
import os

# logging level

logging.basicConfig(level=logging.INFO)

# masks
insured_id_mask_file = 'templates/insured_id-mask.jpg'
birth_date_mask_file = 'templates/birth_date-mask.jpg'
startservdate_1_mask_file = 'templates/startservdate_1-mask.jpg'
endservdate_1_mask_file = 'templates/endservdate_1-mask.jpg'
cpthcpccode_1_mask_file = 'templates/cpthcpccode_1-mask.jpg'
charges_1_mask_file = 'templates/charges_1-mask.jpg'
total_charge_mask_file = 'templates/total_charge-mask.jpg'

# initialization
work_dir = 'work'
mask_height = 2169
mask_width = 1641
template_qrcode_area = 11025 # in pixels
qrcode_to_height_ratio = 20.86
qrcode_to_width_ratio = 15.77
qr_code_area_threshold = 0.1

files = glob('data/test/*.pdf')
# files = ['data/test/18835871_0.pdf']
results = []

for idx, image_file in enumerate(files):

    prefix = datetime.now().strftime("%Y%m%d-%H%M%S")
    logging.info(f"### PROCESSING FILE: {image_file} ({prefix})")

    # initialize output
    result = [image_file.split('/')[-1], "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""]

    # 01 processing setup
    
    input = load_image(image_file, prefix, prefix='input', width=1700)
    input_filename = get_filename(prefix+'_'+image_file.split('/')[-1].split('.')[0], "input")
    cv2.imwrite(input_filename, input)

    # crop from qrcode
    cropped, confidence, area = crop_from_qrcode(input_filename, qrcode_to_height_ratio, qrcode_to_width_ratio) # type: ignore
    if confidence < 0.8:
        logging.info(f"Low confidence record not processed. confidence: {confidence}")
        results.append(result)
        continue
    # check if qrcode is in the right size
    min_area = int(template_qrcode_area * (1 - qr_code_area_threshold))
    max_area = int(template_qrcode_area * (1 + qr_code_area_threshold))
    if area < min_area or area > max_area:
        logging.info(f"QR Code area {area} not in between {min_area} and {max_area}.")
        results.append(result)
        continue
    else:
        logging.info(f"QR Code area {area} in between {min_area} and {max_area}.")
    # cropped = imutils.resize(cropped, height=mask_height, width=mask_width)
    cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    # adjust cropped to the same height as the mask
    cropped_adjusted = np.zeros((mask_height, mask_width), dtype=np.uint8)
    cropped_adjusted[:,:] = 255
    cropped_adjusted[:min(cropped.shape[0], mask_height),:min(cropped.shape[1], mask_width)] = cropped[0:min(cropped.shape[0], mask_height),:min(cropped.shape[1], mask_width)]
    cropped_filename = get_filename(prefix, "cropped")
    cv2.imwrite(cropped_filename, cropped)
    
    ###############################
    # 02 extract insured id
    ###############################
    insured_id = ""

    # apply mask
    insured_id_mask = load_image(insured_id_mask_file, prefix, prefix='insured_id_mask', gray=True )
    insured_id_masked = np.where(insured_id_mask == 0, cropped_adjusted, 255)
    insured_id_masked_filename = get_filename(prefix, "insured_id_masked" )
    cv2.imwrite(insured_id_masked_filename, insured_id_masked)

    # do the ocr
    fr_result = analyze_document_rest(insured_id_masked_filename, "prebuilt-layout", features=['ocr.highResolution'])
    for line in fr_result['pages'][0]['lines']:
        if count_digits(line['content']) >= 6:
            insured_id = line['content'].strip()
            logging.info(f"Insured's id OCR content: {insured_id}")
            insured_id = extract_insured_id(insured_id)
            break
    logging.info(f"Insured's id extracted: {insured_id}")             
    result[1] = insured_id
    
    ###############################
    # 03 extract birth date
    ###############################
    found_birth_date = False
    extracted_birth_date = ""
    valid_birth_date = ""

    # apply mask
    birth_date_mask = load_image(birth_date_mask_file, prefix, prefix='birth_date_mask', gray=True )
    birth_date_masked = np.where(birth_date_mask == 0, cropped_adjusted, 255)
    birth_date_masked_filename = get_filename(prefix, "birth_date_masked" )
    cv2.imwrite(birth_date_masked_filename, birth_date_masked)

    # do a first ocr to get the line with the date    
    fr_result = analyze_document_rest(birth_date_masked_filename, "prebuilt-layout", features=['ocr.highResolution'])
    lines = fr_result['pages'][0]['lines']
    for idx, line in enumerate(lines):
        
        # extract field only when line has at least 4 digits
        if count_digits(line['content']) >= 4:
            if count_digits(line['content']) >= 8:
                ocr_content, polygon = line['content'], line['polygon']
                extracted_birth_date= extract_date(ocr_content)
                if is_valid_date(extracted_birth_date, max_years_old=110): 
                    valid_birth_date = extracted_birth_date
                    found_birth_date = True
                    break                   
            else:
                ocr_content, polygon = concatenate_lines(line, lines)
                logging.info(f"Birth date OCR content: {ocr_content}")
                # first try to extract birth date withour cleaning
                if count_digits(ocr_content) ==6 or count_digits(ocr_content) ==8:
                    extracted_birth_date= extract_date(ocr_content)
                    if is_valid_date(extracted_birth_date, max_years_old=110): 
                        valid_birth_date = extracted_birth_date
                        found_birth_date = True
                        break

            # then try extraction cleaning with different blob sizes
            blob_sizes = [250, 20]
            
            for blob_size in blob_sizes:
                logging.info(f"Cleaning blobs with area < {blob_size} pixels")                    
                
                # extract roi
                birth_date_roi = extract_roi(birth_date_masked_filename, polygon)
                cv2.imwrite(get_filename(prefix, "birth_date_roi_raw" ), birth_date_roi)

                # morphological operations to remove noise
                birth_date_roi = remove_blobs(birth_date_roi, blob_size)
                birth_date_roi = remove_hlines(birth_date_roi)
                birth_date_roi = remove_blobs(birth_date_roi, blob_size)
                birth_date_roi = cv2.cvtColor(birth_date_roi, cv2.COLOR_BGR2GRAY)
                
                # add birth date roi to a new image with the same size as the mask and save it
                birth_date_roi_temp = np.zeros((mask_height, mask_width), dtype=np.uint8)
                birth_date_roi_temp[:,:] = 255
                birth_date_roi_temp[40:birth_date_roi.shape[0]+40,40:birth_date_roi.shape[1]+40] = birth_date_roi[:,:]
                birth_date_roi = birth_date_roi_temp
                birth_date_roi_filename = get_filename(prefix, "birth_date_roi" )
                cv2.imwrite(birth_date_roi_filename, birth_date_roi)
                
                # do a second ocr to get the date after removing noise
                ocr_result2 = analyze_document_rest(birth_date_roi_filename, "prebuilt-layout", features=['ocr.highResolution'])
                lines2 = ocr_result2['pages'][0]['lines']
                for line2 in lines2:    
                    if count_digits(line2['content']) >= 4:
                        content2, polygon2 = concatenate_lines(line2, lines2)
                        extracted_birth_date= extract_date(content2)
                        if is_valid_date(extracted_birth_date, max_years_old=110): 
                            valid_birth_date = extracted_birth_date
                            found_birth_date = True
                            break
                if found_birth_date:
                    break
            break
        if found_birth_date: break
    if not found_birth_date and extracted_birth_date != "": 
        # llm: try to extract birth date
        infered_date = llm_extract_date(extracted_birth_date)
        logging.info(f" LLM results: {infered_date}")           
        if is_valid_date(infered_date, max_years_old=110): 
            found_birth_date = True
            valid_birth_date = infered_date
        else:
            valid_birth_date = ""    
    logging.info(f"Birth date extracted: {valid_birth_date}") 
    result[2] = valid_birth_date

    ###############################
    # 04 extract start serv date 1
    ###############################    
    found_startservdate_1 = False
    extracted_startservdate_1 = ""
    valid_startservdate_1 = ""

    # apply mask
    startservdate_1_mask = load_image(startservdate_1_mask_file, prefix, prefix='startservdate_1_mask', gray=True )
    startservdate_1_masked = np.where(startservdate_1_mask == 0, cropped_adjusted, 255)
    startservdate_1_masked_filename = get_filename(prefix, "startservdate_1_masked" )
    cv2.imwrite(startservdate_1_masked_filename, startservdate_1_masked)

    # do a first ocr to get the line with the date    
    fr_result = analyze_document_rest(startservdate_1_masked_filename, "prebuilt-layout", features=['ocr.highResolution'])
    lines = fr_result['pages'][0]['lines']
    for idx, line in reversed(list(enumerate(lines))): # changed to reversed because the date is usually in the last line
        
        # extract field only when line has at least 4 digits
        if count_digits(line['content']) >= 4:
            if count_digits(line['content']) >= 8:
                ocr_content, polygon = line['content'], line['polygon']
                extracted_startservdate_1= extract_date(ocr_content)
                if is_valid_date(extracted_startservdate_1): 
                    valid_startservdate_1 = extracted_startservdate_1
                    found_startservdate_1 = True
                    break                
            else: 
                # some fields have their value (content) split in two or more lines and we need to concatenate them
                ocr_content, polygon = concatenate_lines(line, lines)
                logging.info(f"Start serv date 1 OCR content: {ocr_content}")
                # first try to extract birth date withour cleaning
                if count_digits(ocr_content) ==6 or count_digits(ocr_content) ==8:
                    extracted_startservdate_1= extract_date(ocr_content)
                    if is_valid_date(extracted_startservdate_1): 
                        valid_startservdate_1 = extracted_startservdate_1
                        found_startservdate_1 = True
                        break

            # then try extraction cleaning with different blob sizes
            blob_sizes = [250, 20]
            
            for blob_size in blob_sizes:
                logging.info(f"Cleaning blobs with area < {blob_size} pixels")                    
                
                # extract roi
                startservdate_1_roi = extract_roi(startservdate_1_masked_filename, polygon)
                cv2.imwrite(get_filename(prefix, "startservdate_1_roi_raw" ), startservdate_1_roi)

                # morphological operations to remove noise
                startservdate_1_roi = remove_blobs(startservdate_1_roi, blob_size)
                startservdate_1_roi = remove_hlines(startservdate_1_roi)
                startservdate_1_roi = remove_blobs(startservdate_1_roi, blob_size)
                startservdate_1_roi = cv2.cvtColor(startservdate_1_roi, cv2.COLOR_BGR2GRAY)
                
                # add start serv 1 date roi to a new image with the same size as the mask and save it
                startservdate_1_roi_temp = np.zeros((mask_height, mask_width), dtype=np.uint8)
                startservdate_1_roi_temp[:,:] = 255
                startservdate_1_roi_temp[40:startservdate_1_roi.shape[0]+40,40:startservdate_1_roi.shape[1]+40] = startservdate_1_roi[:,:]
                startservdate_1_roi = startservdate_1_roi_temp
                startservdate_1_roi_filename = get_filename(prefix, "startservdate_1_roi" )
                cv2.imwrite(startservdate_1_roi_filename, startservdate_1_roi)
                
                # do a second ocr to get the date after removing noise
                ocr_result2 = analyze_document_rest(startservdate_1_roi_filename, "prebuilt-layout", features=['ocr.highResolution'])
                lines2 = ocr_result2['pages'][0]['lines']
                for line2 in lines2:    
                    if count_digits(line2['content']) >= 4:
                        content2, polygon2 = concatenate_lines(line2, lines2)
                        extracted_startservdate_1= extract_date(content2)
                        if is_valid_date(extracted_startservdate_1): 
                            valid_startservdate_1 = extracted_startservdate_1
                            found_startservdate_1 = True
                            break
                if found_startservdate_1:
                    break
            break
        if found_startservdate_1: break
    if not found_startservdate_1 and extracted_startservdate_1 != "": 
        # llm: try to extract start serv 1 date
        infered_date = llm_extract_date(extracted_startservdate_1)
        logging.info(f" LLM results: {infered_date}")           
        if is_valid_date(infered_date): 
            found_startservdate_1 = True
            valid_startservdate_1 = infered_date
        else:
            valid_startservdate_1 = ""    
    logging.info(f"Start serv date 1 extracted: {valid_startservdate_1}") 
    result[3] = valid_startservdate_1

    ###############################
    # 05 extract end serv date 1
    ##############################
    found_endservdate_1 = False
    extracted_endservdate_1 = ""
    valid_endservdate_1 = ""

    # apply mask
    endservdate_1_mask = load_image(endservdate_1_mask_file, prefix, prefix='endservdate_1_mask', gray=True )
    endservdate_1_masked = np.where(endservdate_1_mask == 0, cropped_adjusted, 255)
    endservdate_1_masked_filename = get_filename(prefix, "endservdate_1_masked" )
    cv2.imwrite(endservdate_1_masked_filename, endservdate_1_masked)

    # do a first ocr to get the line with the date    
    fr_result = analyze_document_rest(endservdate_1_masked_filename, "prebuilt-layout", features=['ocr.highResolution'])
    lines = fr_result['pages'][0]['lines']
    for idx, line in reversed(list(enumerate(lines))):

        # extract field only when line has at least 4 digits
        if count_digits(line['content']) >= 4:
            if count_digits(line['content']) >= 8:
                ocr_content, polygon = line['content'], line['polygon']
                extracted_endservdate_1= extract_date(ocr_content)
                if is_valid_date(extracted_endservdate_1): 
                    valid_endservdate_1 = extracted_endservdate_1
                    found_endservdate_1 = True
                    break                
            else:             
                ocr_content, polygon = concatenate_lines(line, lines)
                logging.info(f"End serv date 1 OCR content: {ocr_content}")
                # first try to extract birth date withour cleaning
                if count_digits(ocr_content) ==6 or count_digits(ocr_content) ==8:
                    extracted_endservdate_1= extract_date(ocr_content)
                    if is_valid_date(extracted_endservdate_1): 
                        valid_endservdate_1 = extracted_endservdate_1
                        found_endservdate_1 = True
                        break

            # then try extraction cleaning with different blob sizes
            blob_sizes = [250, 20]
            
            for blob_size in blob_sizes:
                logging.info(f"Cleaning blobs with area < {blob_size} pixels")                    
                
                # extract roi
                endservdate_1_roi = extract_roi(endservdate_1_masked_filename, polygon)
                cv2.imwrite(get_filename(prefix, "endservdate_1_roi_raw" ), endservdate_1_roi)

                # morphological operations to remove noise
                endservdate_1_roi = remove_blobs(endservdate_1_roi, blob_size)
                endservdate_1_roi = remove_hlines(endservdate_1_roi)
                endservdate_1_roi = remove_blobs(endservdate_1_roi, blob_size)
                endservdate_1_roi = cv2.cvtColor(endservdate_1_roi, cv2.COLOR_BGR2GRAY)
                
                # add end serv 1 date roi to a new image with the same size as the mask and save it
                endservdate_1_roi_temp = np.zeros((mask_height, mask_width), dtype=np.uint8)
                endservdate_1_roi_temp[:,:] = 255
                endservdate_1_roi_temp[40:endservdate_1_roi.shape[0]+40,40:endservdate_1_roi.shape[1]+40] = endservdate_1_roi[:,:]
                endservdate_1_roi = endservdate_1_roi_temp
                endservdate_1_roi_filename = get_filename(prefix, "endservdate_1_roi" )
                cv2.imwrite(endservdate_1_roi_filename, endservdate_1_roi)
                
                # do a second ocr to get the date after removing noise
                ocr_result2 = analyze_document_rest(endservdate_1_roi_filename, "prebuilt-layout", features=['ocr.highResolution'])
                lines2 = ocr_result2['pages'][0]['lines']
                for line2 in lines2:    
                    if count_digits(line2['content']) >= 4:
                        content2, polygon2 = concatenate_lines(line2, lines2)
                        extracted_endservdate_1= extract_date(content2)
                        if is_valid_date(extracted_endservdate_1): 
                            valid_endservdate_1 = extracted_endservdate_1
                            found_endservdate_1 = True
                            break
                if found_endservdate_1:
                    break
            break
        if found_endservdate_1: break
    if not found_endservdate_1 and extracted_endservdate_1 != "": 
        # llm: try to extract end serv 1 date
        infered_date = llm_extract_date(extracted_endservdate_1)
        logging.info(f" LLM results: {infered_date}")           
        if is_valid_date(infered_date): 
            found_endservdate_1 = True
            valid_endservdate_1 = infered_date
        else:
            valid_endservdate_1 = ""    
    logging.info(f"End serv date 1 extracted: {valid_endservdate_1}") 
    result[4] = valid_endservdate_1

    ###############################
    # 06 extract cpth cpc code 1
    ###############################
    cpthcpccode_1 = ""

    # apply mask
    cpthcpccode_1_mask = load_image(cpthcpccode_1_mask_file, prefix, prefix='cpthcpccode_1_mask', gray=True )
    cpthcpccode_1_masked = np.where(cpthcpccode_1_mask == 0, cropped_adjusted, 255)
    cpthcpccode_1_masked_filename = get_filename(prefix, "cpthcpccode_1_masked" )
    cv2.imwrite(cpthcpccode_1_masked_filename, cpthcpccode_1_masked)

    # do the ocr
    fr_result = analyze_document_rest(cpthcpccode_1_masked_filename, "prebuilt-layout", features=['ocr.highResolution'])
    for line in fr_result['pages'][0]['lines']:
        if count_digits(line['content']) >= 4:
            cpthcpccode_1 = line['content'].strip()
            logging.info(f"Cpt's OCR content: {cpthcpccode_1}")
            cpthcpccode_1 = extract_cpthcpccode(cpthcpccode_1)[-5:]
            break
    logging.info(f"Cpt code extracted: {cpthcpccode_1}")             
    result[6] = cpthcpccode_1

    ###############################
    # 07 extract charges 1
    ###############################
    charges_1 = ""

    # apply mask
    charges_1_mask = load_image(charges_1_mask_file, prefix, prefix='charges_1_mask', gray=True )
    charges_1_masked = np.where(charges_1_mask == 0, cropped_adjusted, 255)
    charges_1_masked_filename = get_filename(prefix, "charges_1_masked" )
    cv2.imwrite(charges_1_masked_filename, charges_1_masked)   
    blob_sizes = [40]
    attempt = 1
    for blob_size in blob_sizes:
        logging.info(f"Charges' 1 attempt {attempt} blob size {blob_size}")
        charges_1_masked_temp = remove_blobs(charges_1_masked, blob_size)
        charges_1_masked_temp = remove_hlines(charges_1_masked_temp)
        charges_1_masked_temp = remove_blobs(charges_1_masked_temp, blob_size)
        charges_1_masked_temp_filename = get_filename(prefix, "charges_1_masked_temp" )
        cv2.imwrite(charges_1_masked_temp_filename, charges_1_masked_temp)
        # do the ocr
        fr_result = analyze_document_rest(charges_1_masked_temp_filename, "prebuilt-layout", features=['ocr.highResolution'])
        lines = fr_result['pages'][0]['lines']
        for line in lines:
            if count_digits(line['content']) >= 3:
                charges_1 = line['content'].strip()
                logging.info(f"Charges' 1 OCR content: {charges_1}")
                charges_1 = extract_charges(charges_1)
                break
        if charges_1 != "": break
        attempt += 1
    if charges_1 == "":
        logging.info(f"Charges' 1 attempt {attempt} no cleanning")
        fr_result = analyze_document_rest(charges_1_masked_filename, "prebuilt-layout", features=['ocr.highResolution'])
        lines = fr_result['pages'][0]['lines']
        for line in lines:
            if count_digits(line['content']) >= 3:
                charges_1 = line['content'].strip()
                logging.info(f"Charges' 1 OCR content: {charges_1}")
                charges_1 = extract_charges(charges_1)
                break
    logging.info(f"Charges' 1 extracted: {charges_1}")             
    result[7] = charges_1


    ###############################
    # 08 extract total charge
    ###############################
    total_charge = ""

    # apply mask
    total_charge_mask = load_image(total_charge_mask_file, prefix, prefix='total_charge_mask', gray=True )
    total_charge_masked = np.where(total_charge_mask == 0, cropped_adjusted, 255)
    total_charge_masked_filename = get_filename(prefix, "total_charge_masked" )
    cv2.imwrite(total_charge_masked_filename, total_charge_masked)   
    blob_sizes = [40]
    attempt = 1
    for blob_size in blob_sizes:
        logging.info(f"Total charge attempt {attempt} blob size {blob_size}")
        total_charge_masked_temp = remove_blobs(total_charge_masked, blob_size)
        total_charge_masked_temp = remove_hlines(total_charge_masked_temp)
        total_charge_masked_temp = remove_blobs(total_charge_masked_temp, blob_size)
        total_charge_masked_temp_filename = get_filename(prefix, "total_charge_masked_temp" )
        cv2.imwrite(total_charge_masked_temp_filename, total_charge_masked_temp)
        # do the ocr
        fr_result = analyze_document_rest(total_charge_masked_temp_filename, "prebuilt-layout", features=['ocr.highResolution'])
        lines = fr_result['pages'][0]['lines']
        for line in lines:
            if count_digits(line['content']) >= 3:
                total_charge = line['content'].strip()
                logging.info(f"Total charge OCR content: {total_charge}")
                total_charge = extract_charges(total_charge)
                break
        if total_charge != "": break
        attempt += 1
    if total_charge == "":
        logging.info(f"Total charge attempt {attempt} no cleanning")
        fr_result = analyze_document_rest(total_charge_masked_filename, "prebuilt-layout", features=['ocr.highResolution'])
        lines = fr_result['pages'][0]['lines']
        for line in lines:
            if count_digits(line['content']) >= 3:
                total_charge = line['content'].strip()
                logging.info(f"Total charge OCR content: {total_charge}")
                total_charge = extract_charges(total_charge)
                break
    logging.info(f"Total charge extracted: {total_charge}")             
    result[28] = total_charge

    # 0n extract the next field and so on ...
    # TODO

    results.append(result)

# save results
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
if not os.path.exists(work_dir): os.makedirs(work_dir)
output_file = os.path.join(work_dir, f'{timestamp}.csv')
with open(output_file, "w", newline="") as f:
    writer = csv.writer(f)
    header = ["fileName", "patientInsuredId", "patientBirthday", "startServDate_1", "endServDate_1", "qty_1", "cptHcpcCode_1", "charges_1", "startServDate_2", "endServDate_2", "qty_2", "cptHcpcCode_2", "charges_2", "startServDate_3", "endServDate_3", "qty_3", "cptHcpcCode_3", "charges_3", "startServDate_4", "endServDate_4", "qty_4", "cptHcpcCode_4", "charges_4", "startServDate_5", "endServDate_5", "qty_5", "cptHcpcCode_5", "charges_5", "patientZipcode", "total_charge"]
    writer.writerow(header)
    writer.writerows(results)

logging.info(f"Results file: {output_file}") 
logging.info(f"Done")
